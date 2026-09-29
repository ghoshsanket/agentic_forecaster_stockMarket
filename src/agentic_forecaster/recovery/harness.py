"""STAGE 0 sanity harness: can the model learn at all?

Two cheap experiments that must pass BEFORE any expensive search:

**Tiny overfit.**  Train on a very small TRAIN-only subset long enough that
training accuracy should approach 1.0 and training loss should approach 0.  If
the Attention-LSTM cannot deliberately overfit a tiny subset, then something is
wrong in the model, the optimizer, the target alignment, the gradients or the
scaling - and every later number would be uninterpretable.

**Shuffled-label control.**  Shuffle TRAIN labels only and re-run.  Validation
performance must fall to approximately chance.  A model that still looks good on
shuffled labels is not using the inputs, which invalidates any recovered result.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from agentic_forecaster.recovery.firewall import assert_pre_test_dates, firewall_guard


@dataclass
class SanityResult:
    """Outcome of a single STAGE 0 experiment."""

    name: str
    passed: bool
    n_train: int
    n_val: int
    train_loss: float | None = None
    train_accuracy: float | None = None
    validation_accuracy: float | None = None
    validation_brier: float | None = None
    majority_validation_accuracy: float | None = None
    chance_validation_accuracy: float | None = None
    best_epoch: int | None = None
    epochs_run: int | None = None
    best_train_accuracy: float | None = None
    best_train_loss: float | None = None
    diagnostics: dict = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "n_train": self.n_train,
            "n_val": self.n_val,
            "train_loss": self.train_loss,
            "train_accuracy": self.train_accuracy,
            "validation_accuracy": self.validation_accuracy,
            "validation_brier": self.validation_brier,
            "majority_validation_accuracy": self.majority_validation_accuracy,
            "chance_validation_accuracy": self.chance_validation_accuracy,
            "best_epoch": self.best_epoch,
            "epochs_run": self.epochs_run,
            "best_train_accuracy": self.best_train_accuracy,
            "best_train_loss": self.best_train_loss,
            "diagnostics": self.diagnostics,
            "notes": self.notes,
        }


def _fit_and_score(X_train, y_train, X_val, y_val, *, epochs, seed,
                   hidden_size, num_layers, dropout, learning_rate,
                   batch_size, weight_decay, patience,
                   class_weighting="none") -> dict:
    """Train a small attention-LSTM and return train/val diagnostics."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    from agentic_forecaster.models.attention_lstm import AttentionLSTM

    torch.manual_seed(seed)
    np.random.seed(seed)

    Xtr = torch.tensor(np.asarray(X_train), dtype=torch.float32)
    ytr = torch.tensor(np.asarray(y_train), dtype=torch.float32).reshape(-1, 1)
    Xva = torch.tensor(np.asarray(X_val), dtype=torch.float32)
    yva = torch.tensor(np.asarray(y_val), dtype=torch.float32).reshape(-1, 1)

    input_dim = Xtr.shape[-1]
    model = AttentionLSTM(
        input_size=input_dim,
        hidden_size=int(hidden_size),
        num_layers=int(num_layers),
        dropout=float(dropout),
    )
    optimiser = torch.optim.Adam(
        model.parameters(), lr=learning_rate, betas=(0.9, 0.999),
        weight_decay=weight_decay)

    if class_weighting == "pos_weight":
        # pos_weight is derived from TRAIN labels only, never validation/test.
        pos = float(np.sum(y_train == 1))
        neg = float(np.sum(y_train == 0))
        pos_weight = torch.tensor([[(neg / pos) if pos > 0 else 1.0]],
                                  dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    def _flat(z: torch.Tensor) -> torch.Tensor:
        return z.reshape(-1)

    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)
    best_val = float("inf")
    best_epoch = None
    best_state = None
    stale = 0
    # The overfit question is "can the model FIT ITS TRAINING DATA", which is the
    # best train accuracy reached during training - not the train accuracy at the
    # best-VALIDATION epoch, which can be an early, under-fit checkpoint.
    best_train_accuracy = 0.0
    best_train_loss = float("inf")
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            optimiser.zero_grad()
            out = _flat(model(xb))
            loss = criterion(out, _flat(yb))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            losses.append(float(loss.detach()))

        model.eval()
        with torch.no_grad():
            val_logits = _flat(model(Xva))
            yva_f = _flat(yva)
            val_loss = float(nn.functional.binary_cross_entropy_with_logits(
                val_logits, yva_f))
            tr_logits = _flat(model(Xtr))
            ytr_f = _flat(ytr)
            tr_loss = float(nn.functional.binary_cross_entropy_with_logits(
                tr_logits, ytr_f))
            tr_acc = float(((tr_logits >= 0).float() == ytr_f).float().mean())
            val_acc = float(((val_logits >= 0).float() == yva_f).float().mean())
            val_proba = torch.sigmoid(val_logits).numpy().reshape(-1)

        best_train_accuracy = max(best_train_accuracy, tr_acc)
        best_train_loss = min(best_train_loss, tr_loss)
        history.append({"epoch": epoch, "train_loss": tr_loss,
                        "validation_loss": val_loss, "train_accuracy": tr_acc,
                        "validation_accuracy": val_acc})
        if val_loss < best_val - 1e-6:
            best_val, best_epoch, stale = val_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        tr_logits = _flat(model(Xtr))
        final_tr_loss = float(nn.functional.binary_cross_entropy_with_logits(
            tr_logits, _flat(ytr)))
        final_tr_acc = float(((tr_logits >= 0).float() == _flat(ytr)).float().mean())
        val_proba = torch.sigmoid(_flat(model(Xva))).numpy().reshape(-1)

    yv = np.asarray(y_val).astype(int)
    return {
        "train_loss": final_tr_loss,
        "train_accuracy": final_tr_acc,
        "validation_accuracy": float(((val_proba >= 0.5).astype(int) == yv).mean()),
        "validation_brier": float(np.mean((val_proba - yv) ** 2)),
        "majority_validation_accuracy": float(
            max(yv.mean(), 1.0 - yv.mean())),
        "chance_validation_accuracy": 0.5,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "best_train_accuracy": best_train_accuracy,
        "best_train_loss": best_train_loss,
        "history": history,
    }


def run_tiny_overfit(X_train, y_train, X_val, y_val, *, val_dates=None,
                     seed=42, hidden_size=64, num_layers=2, dropout=0.0,
                     learning_rate=1e-3, batch_size=64, weight_decay=0.0,
                     epochs=300, patience=1000, max_train_rows=256,
                     class_weighting="none", min_train_accuracy=0.95,
                     name="tiny_overfit") -> SanityResult:
    """Prove the model can deliberately overfit a tiny TRAIN-only subset.

    Only TRAIN rows are subsampled.  The validation split is left intact so the
    result is comparable with the shuffled control.
    """
    if val_dates is not None:
        assert_pre_test_dates(val_dates, where="tiny_overfit/validation", search=True)

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    n = len(X_train)
    if n > max_train_rows:
        # take the most recent rows so the subset is still in-distribution
        X_train = X_train[-max_train_rows:]
        y_train = y_train[-max_train_rows:]

    with firewall_guard(True):
        out = _fit_and_score(
            X_train, y_train, X_val, y_val, epochs=epochs, seed=seed,
            hidden_size=hidden_size, num_layers=num_layers, dropout=dropout,
            learning_rate=learning_rate, batch_size=batch_size,
            weight_decay=weight_decay, patience=patience,
            class_weighting=class_weighting)

    passed = bool(out["best_train_accuracy"] >= min_train_accuracy)
    res = SanityResult(
        name=name, passed=passed, n_train=len(X_train), n_val=len(X_val),
        train_loss=out["train_loss"], train_accuracy=out["train_accuracy"],
        validation_accuracy=out["validation_accuracy"],
        validation_brier=out["validation_brier"],
        majority_validation_accuracy=out["majority_validation_accuracy"],
        chance_validation_accuracy=out["chance_validation_accuracy"],
        best_epoch=out["best_epoch"], epochs_run=out["epochs_run"],
        best_train_accuracy=out["best_train_accuracy"],
        best_train_loss=out["best_train_loss"],
        diagnostics={"history": out["history"][-20:],
                     "min_train_accuracy_required": min_train_accuracy,
                     "max_train_rows": max_train_rows},
        notes=("PASS: the model can overfit a tiny TRAIN subset, so the model, "
               "optimiser, target alignment, gradients and scaling are sound."
               if passed else
               "STOP: the model could NOT overfit a tiny TRAIN subset. Investigate "
               "model / optimizer / target alignment / gradients / scaling before "
               "running any search."))
    return res


def run_shuffled_label_control(X_train, y_train, X_val, y_val, *,
                               val_dates=None, seed=42, hidden_size=64,
                               num_layers=2, dropout=0.0, learning_rate=1e-3,
                               batch_size=64, weight_decay=0.0, epochs=100,
                               patience=10, class_weighting="none",
                               name="shuffled_label_control") -> SanityResult:
    """Shuffle TRAIN labels only; validation performance must fall to chance."""
    if val_dates is not None:
        assert_pre_test_dates(val_dates, where="shuffled_control/validation", search=True)

    rng = np.random.default_rng(seed)
    y_scrambled = np.array(y_train, copy=True)
    rng.shuffle(y_scrambled)

    with firewall_guard(True):
        out = _fit_and_score(
            X_train, y_scrambled, X_val, y_val, epochs=epochs, seed=seed,
            hidden_size=hidden_size, num_layers=num_layers, dropout=dropout,
            learning_rate=learning_rate, batch_size=batch_size,
            weight_decay=weight_decay, patience=patience,
            class_weighting=class_weighting)

    # Chance band: within 3 standard errors of 0.5 for a binomial.
    n_val = len(y_val)
    se = float(np.sqrt(0.25 / max(n_val, 1)))
    near_chance = bool(abs(out["validation_accuracy"] - 0.5) <= 3 * se)
    res = SanityResult(
        name=name, passed=near_chance, n_train=len(X_train), n_val=n_val,
        train_loss=out["train_loss"], train_accuracy=out["train_accuracy"],
        validation_accuracy=out["validation_accuracy"],
        validation_brier=out["validation_brier"],
        majority_validation_accuracy=out["majority_validation_accuracy"],
        chance_validation_accuracy=0.5, best_epoch=out["best_epoch"],
        epochs_run=out["epochs_run"],
        diagnostics={"shuffle_seed": seed, "chance_se": se,
                     "chance_band_3se": [0.5 - 3 * se, 0.5 + 3 * se]},
        notes=("PASS: shuffled TRAIN labels give chance validation accuracy, so the "
               "model is genuinely using the inputs."
               if near_chance else
               "STOP: shuffled TRAIN labels did NOT give chance validation accuracy. "
               "The model is not using the inputs; diagnose before searching."))
    return res


def run_real_label_reference(X_train, y_train, X_val, y_val, *, val_dates=None,
                             seed=42, hidden_size=64, num_layers=2, dropout=0.0,
                             learning_rate=1e-3, batch_size=32, weight_decay=0.0,
                             epochs=100, patience=10,
                             name="real_label_reference") -> SanityResult:
    """Train a NORMAL model on the full TRAIN split to give the control a
    fair opponent.

    The shuffled-label control is only meaningful if it is compared against a
    properly trained real-label model, not against the deliberately-overfit
    probe. This function is that opponent: same architecture and optimizer as
    the control, real labels, normal early-stopped budget.
    """
    if val_dates is not None:
        assert_pre_test_dates(val_dates, where="real_reference/validation", search=True)
    with firewall_guard(True):
        out = _fit_and_score(
            X_train, y_train, X_val, y_val, epochs=epochs, seed=seed,
            hidden_size=hidden_size, num_layers=num_layers, dropout=dropout,
            learning_rate=learning_rate, batch_size=batch_size,
            weight_decay=weight_decay, patience=patience)
    return SanityResult(
        name=name, passed=True, n_train=len(X_train), n_val=len(X_val),
        train_loss=out["train_loss"], train_accuracy=out["train_accuracy"],
        validation_accuracy=out["validation_accuracy"],
        validation_brier=out["validation_brier"],
        majority_validation_accuracy=out["majority_validation_accuracy"],
        chance_validation_accuracy=0.5, best_epoch=out["best_epoch"],
        epochs_run=out["epochs_run"],
        best_train_accuracy=out["best_train_accuracy"],
        best_train_loss=out["best_train_loss"],
        diagnostics={"shuffle_seed": None},
        notes="Real-label reference model for the shuffled-label comparison.")


def save_stage0(results: list[SanityResult], out_dir: Path) -> dict:
    """Write ``tiny_overfit.json`` / ``shuffled_label_control.json`` + summary."""
    from datetime import UTC, datetime

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "stage": "STAGE_0",
        "all_passed": all(r.passed for r in results),
        "results": {r.name: r.to_dict() for r in results},
    }
    (out_dir / "stage0_summary.json").write_text(json.dumps(payload, indent=2))
    for r in results:
        (out_dir / f"{r.name}.json").write_text(
            json.dumps(r.to_dict(), indent=2))
    return payload
