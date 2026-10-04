"""V2-F: genuine but deliberately LIGHTWEIGHT meta-learning.

Label
-----
This module implements ``REPTILE_STYLE_HEAD_ADAPTER``.  It is NOT MAML and it is
NOT FiLM:

* FiLM (V2-E) is conditional adaptation -- gamma and beta are a deterministic
  function of the identity/regime embeddings, learned jointly with everything
  else.
* Reptile here is META-LEARNING -- a task-specific parameter set is obtained by
  taking inner gradient steps on that task's support set, and the shared
  meta-initialisation is then moved towards those adapted parameters.

WHAT IS AND IS NOT ADAPTED
---------------------------
Adapted (inner loop):  the residual adapter block and the three task heads.
Frozen (always):       the LSTM, the Transformer, the context projection, the
                       positional encoding, the ticker/sector/regime embeddings,
                       the fusion MLP, the FiLM parameters.

No third-party meta-learning dependency is used: PyTorch only.

META TASK DEFINITION
--------------------
A meta task is NOT "one stock = one task".  It is

    security + chronological historical episode = one task

with ``support = the preceding 120 supervised samples`` and
``query = the immediately following 20 supervised samples``, stride 20.  Support
dates always precede query dates, no random mixing is possible, and no episode
may cross a fold boundary.  Episodes are built from TRAIN samples only, so no
validation, lockbox or 2022+ label can enter an episode.

At validation time each security is adapted from the last 120 labelled TRAIN
samples immediately preceding the validation window, using 3 support updates and
no validation label.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch import nn

from .dataset import SampleTable, V2SequenceDataset
from .firewall import (
    assert_no_lockbox_targets,
    assert_no_paper_test_targets,
    assert_pre_covid_dates,
)
from .losses import MultiTaskWeights, V2MultiTaskLoss
from .model import ContextualLSTMTransformer

logger = logging.getLogger("agentic_forecaster.v2.meta")

#: Honest, machine-readable label of this adaptation scheme.
META_LABEL = "REPTILE_STYLE_HEAD_ADAPTER"

#: Not implemented on purpose, and stated so it cannot be claimed later.
NOT_MAML = "NOT_MAML_FULL_NETWORK_INNER_LOOP_NOT_IMPLEMENTED"


@dataclass(frozen=True)
class MetaConfig:
    """Fixed initial meta-learning settings (not searched in this programme)."""

    support_size: int = 120
    query_size: int = 20
    stride: int = 20
    inner_steps: int = 3
    inner_lr: float = 1e-3
    meta_step_size: float = 0.10
    episodes_per_meta_batch: int = 8
    meta_epochs: int = 5
    validation_support_size: int = 120
    validation_inner_steps: int = 3

    def to_dict(self) -> dict:
        return {
            "label": META_LABEL,
            "not": NOT_MAML,
            "support_size": self.support_size,
            "query_size": self.query_size,
            "stride": self.stride,
            "inner_steps": self.inner_steps,
            "inner_lr": self.inner_lr,
            "meta_step_size": self.meta_step_size,
            "episodes_per_meta_batch": self.episodes_per_meta_batch,
            "meta_epochs": self.meta_epochs,
            "validation_support_size": self.validation_support_size,
            "validation_inner_steps": self.validation_inner_steps,
            "adapted": "residual adapter + task heads",
            "frozen": "LSTM + Transformer + context encoder + embeddings",
        }


@dataclass
class MetaEpisode:
    """One chronological support/query task for a single security."""

    ticker: str
    support_indices: list[int]
    query_indices: list[int]
    support_origin_dates: list[pd.Timestamp] = field(default_factory=list)
    query_origin_dates: list[pd.Timestamp] = field(default_factory=list)

    def validate(self) -> None:
        if not self.support_indices or not self.query_indices:
            raise ValueError("a meta episode needs a non-empty support and query set")
        if max(self.support_origin_dates) >= min(self.query_origin_dates):
            raise ValueError(
                f"{self.ticker}: a query date precedes a support date; episodes must "
                "be chronological"
            )


def build_meta_episodes(samples: SampleTable, *, config: MetaConfig,
                        train_mask: np.ndarray, max_episodes: int | None = None,
                        seed: int = 42, where: str = "meta episodes",
                        final_allowed_date: str | None = None) -> list[MetaEpisode]:
    """Chronological episodes inside the TRAIN window only.

    Episodes are enumerated per security over the ordered TRAIN samples and then
    sampled with a fixed seed, so the episode set is reproducible.
    """
    frame = samples.frame.loc[train_mask].sort_values(["ticker", "origin_date"])
    episodes: list[MetaEpisode] = []
    total = config.support_size + config.query_size

    for ticker, group in frame.groupby("ticker", sort=True):
        # ``samples.frame`` carries a RangeIndex, so index labels are positions and
        # can be used directly with ``.iloc`` when the side is materialised.
        positions = group.index.to_numpy()
        origins = pd.to_datetime(group["origin_date"]).to_numpy()
        n = len(group)
        start = 0
        while start + total <= n:
            support = positions[start:start + config.support_size]
            query = positions[start + config.support_size:start + total]
            episode = MetaEpisode(
                ticker=str(ticker),
                support_indices=[int(i) for i in support],
                query_indices=[int(i) for i in query],
                support_origin_dates=[pd.Timestamp(d) for d in origins[
                    start:start + config.support_size]],
                query_origin_dates=[pd.Timestamp(d) for d in origins[
                    start + config.support_size:start + total]],
            )
            episode.validate()
            episodes.append(episode)
            start += config.stride

    if max_episodes and len(episodes) > max_episodes:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(episodes), size=max_episodes, replace=False)
        episodes = [episodes[int(i)] for i in sorted(chosen)]

    # Both firewalls apply to every episode: an episode must not contain a 2021
    # lockbox target (episodes are TRAIN-only) nor a 2022+ target (forbidden for
    # the whole programme).  The check is on TARGET dates, because a target date
    # is what identifies a supervised outcome.
    covered = sorted({index for episode in episodes
                      for index in episode.support_indices + episode.query_indices})
    covered_frame = samples.frame.loc[covered]
    assert_no_paper_test_targets(covered_frame["target_date"], where=f"{where}/target")
    assert_no_paper_test_targets(covered_frame["origin_date"], where=f"{where}/origin")
    assert_no_lockbox_targets(covered_frame["target_date"], where=f"{where}/target")
    if final_allowed_date is not None:
        # PRE-COVID regime: no episode may reach past the boundary, and the
        # rejection happens here rather than as a metric filter later
        assert_pre_covid_dates(origin_dates=covered_frame["origin_date"],
                               target_dates=covered_frame["target_date"],
                               final_allowed_date=final_allowed_date,
                               where=f"{where}/target")
    return episodes


def episode_frame(samples: SampleTable, episode: MetaEpisode, indices: list[int]) -> pd.DataFrame:
    """The sample rows of one episode side, as a frame the Dataset accepts."""
    return samples.frame.iloc[indices].reset_index(drop=True)


def _clone_adaptable(model: ContextualLSTMTransformer) -> dict[str, torch.Tensor]:
    """Clone the adaptable (adapter + head) parameters for an inner loop."""
    names = set(model.adaptable_parameter_names())
    return {name: parameter.detach().clone()
            for name, parameter in model.named_parameters() if name in names}


def _load_adaptable(model: ContextualLSTMTransformer,
                    state: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in state:
                parameter.copy_(state[name].to(parameter.device))


def _inner_update(model: ContextualLSTMTransformer, criterion: V2MultiTaskLoss,
                  loader, device: torch.device, *, steps: int, lr: float,
                  meta_init: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], list]:
    """Inner loop: clone, adapt on support, return the adapted parameters.

    The shared encoder is frozen throughout: only the adapter and heads receive
    gradients.  The returned parameters are the task-specific ones; the caller
    decides what to do with them (validation) or how to fold them back into the
    meta-initialisation (training).
    """
    task_state = _clone_adaptable(model)
    _load_adaptable(model, meta_init)
    model.set_encoder_frozen(True)
    for parameter in model.parameters():
        parameter.requires_grad_(name_is_adaptable(model, parameter))

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("no adaptable parameter is trainable; meta-learning cannot run")
    optimiser = torch.optim.SGD(trainable, lr=lr)
    losses: list[float] = []
    model.train()
    for _ in range(steps):
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch.items()
                      if k in ("stock_sequence", "context_sequence", "regime_vector",
                               "ticker_id", "sector_id")}
            targets = {k: batch[k].to(device) for k in ("y_direction", "y_return", "y_rank")
                       if k in batch}
            optimiser.zero_grad(set_to_none=True)
            outputs = model(inputs["stock_sequence"], inputs["ticker_id"],
                            context_sequence=inputs["context_sequence"],
                            sector_id=inputs["sector_id"],
                            regime_vector=inputs["regime_vector"])
            loss, _ = criterion(outputs, targets)
            loss.backward()
            optimiser.step()
            losses.append(float(loss.detach()))
            break                      # one inner batch per inner step
    adapted = _clone_adaptable(model)
    _load_adaptable(model, task_state)   # restore the pre-adaptation parameters
    return adapted, losses


def name_is_adaptable(model: ContextualLSTMTransformer, parameter: nn.Parameter) -> bool:
    """True when ``parameter`` belongs to the adaptable (adapter/head) set."""
    for name, candidate in model.named_parameters():
        if candidate is parameter:
            return name in set(model.adaptable_parameter_names())
    return False


@torch.no_grad()
def _query_loss(model: ContextualLSTMTransformer, criterion: V2MultiTaskLoss,
                loader, device: torch.device) -> float:
    model.eval()
    total, n_batches = 0.0, 0
    for batch in loader:
        inputs = {k: v.to(device) for k, v in batch.items()
                  if k in ("stock_sequence", "context_sequence", "regime_vector",
                           "ticker_id", "sector_id")}
        targets = {k: batch[k].to(device) for k in ("y_direction", "y_return", "y_rank")
                   if k in batch}
        outputs = model(inputs["stock_sequence"], inputs["ticker_id"],
                        context_sequence=inputs["context_sequence"],
                        sector_id=inputs["sector_id"],
                        regime_vector=inputs["regime_vector"])
        loss, _ = criterion(outputs, targets)
        total += float(loss.detach())
        n_batches += 1
    return total / max(n_batches, 1)


class ReptileMetaTrainer:
    """Reptile-style adaptation of the adapter and heads.

    One meta step, per the standard Reptile formulation:

        theta_init <- theta_init + eps * mean_tasks(theta_task - theta_init)

    where ``theta_task`` is obtained by ``inner_steps`` gradient steps on that
    task's chronological support set, starting from ``theta_init``.
    """

    def __init__(self, model: ContextualLSTMTransformer, config: MetaConfig, *,
                 loss_weights: MultiTaskWeights | None = None,
                 device: torch.device | str = "cpu", batch_size: int = 32,
                 final_allowed_date: str | None = None) -> None:
        self.model = model
        self.config = config
        self.final_allowed_date = final_allowed_date
        self.device = torch.device(device)
        self.criterion = V2MultiTaskLoss(loss_weights)
        self.batch_size = batch_size
        self.meta_init: dict[str, torch.Tensor] = _clone_adaptable(model)

    # ------------------------------------------------------------------ meta

    def fit(self, samples: SampleTable, train_mask: np.ndarray, *, seed: int = 42,
            where: str = "meta training") -> dict:
        """Run the Reptile meta-training over chronological TRAIN episodes.

        Each meta epoch is ONE meta batch of ``episodes_per_meta_batch``
        episodes, drawn without replacement from the available chronological TRAIN
        episodes with a fixed seed.  That keeps the stage genuinely lightweight:
        5 meta epochs x 8 episodes x 3 inner steps = 120 gradient steps in total,
        against the shared encoder's tens of epochs of full-batch training.
        """
        episodes = build_meta_episodes(samples, config=self.config, train_mask=train_mask,
                                       seed=seed, where=where,
                                       final_allowed_date=self.final_allowed_date)
        if not episodes:
            raise RuntimeError("no chronological meta episode could be built from TRAIN data")
        rng = np.random.default_rng(seed)
        history: list[dict] = []
        per_epoch = min(self.config.episodes_per_meta_batch, len(episodes))

        for epoch in range(1, self.config.meta_epochs + 1):
            chosen = rng.choice(len(episodes), size=per_epoch, replace=False)
            pre_losses: list[float] = []
            post_losses: list[float] = []
            deltas: list[float] = []
            task_parameters = []
            for episode_id in sorted(int(c) for c in chosen):
                episode = episodes[episode_id]
                adapted, support_losses = self._adapt_episode(
                    samples, episode, self.meta_init)
                task_parameters.append(adapted)
                if support_losses:
                    pre_losses.append(support_losses[0])
                    post_losses.append(support_losses[-1])
                deltas.append(_parameter_distance(self.meta_init, adapted))

            # Reptile movement of the meta initialisation towards the mean task
            # parameter set of this meta batch.
            for name, tensor in self.meta_init.items():
                mean_task = torch.stack(
                    [task[name].to(tensor.device).float() for task in task_parameters]
                ).mean(dim=0)
                tensor.add_(self.config.meta_step_size * (mean_task - tensor.float()))
            _load_adaptable(self.model, self.meta_init)
            history.append({
                "meta_epoch": epoch,
                "n_episodes": len(task_parameters),
                "mean_support_loss_before": float(np.mean(pre_losses)) if pre_losses else None,
                "mean_support_loss_after": float(np.mean(post_losses)) if post_losses else None,
                "mean_parameter_delta": float(np.mean(deltas)) if deltas else 0.0,
            })
            logger.info("meta epoch %d: %d episodes, mean |delta| = %.3e", epoch,
                        len(task_parameters), history[-1]["mean_parameter_delta"])

        _load_adaptable(self.model, self.meta_init)
        return {
            "label": META_LABEL,
            "config": self.config.to_dict(),
            "n_episodes_available": len(episodes),
            "n_tickers": len({e.ticker for e in episodes}),
            "history": history,
            "meta_initialization": self.meta_init,
        }

    def _adapt_episode(self, samples: SampleTable, episode: MetaEpisode,
                       meta_init: dict[str, torch.Tensor]) -> tuple[dict, list[float]]:
        support_frame = episode_frame(samples, episode, episode.support_indices)
        support_loader = _loader_for(samples, support_frame, self.batch_size)
        return _inner_update(
            self.model, self.criterion, support_loader, self.device,
            steps=self.config.inner_steps, lr=self.config.inner_lr, meta_init=meta_init)

    # ---------------------------------------------------------- validation

    def adapt_for_prediction(self, samples: SampleTable, *, support_frame: pd.DataFrame,
                             ticker: str) -> dict[str, torch.Tensor]:
        """Adapt the adapter/heads for ONE security before predicting it.

        ``support_frame`` must contain TRAIN samples only.  The caller is
        responsible for that (see :func:`validation_support_frames`); no
        validation label is ever read here.
        """
        loader = _loader_for(samples, support_frame, self.batch_size)
        adapted, _ = _inner_update(
            self.model, self.criterion, loader, self.device,
            steps=self.config.validation_inner_steps, lr=self.config.inner_lr,
            meta_init=self.meta_init)
        return adapted

    def score_with_adaptation(self, samples: SampleTable, *, support_frame: pd.DataFrame,
                              query_frame: pd.DataFrame, ticker: str) -> dict:
        """Adapt on SUPPORT, then score the QUERY set: the leakage-free protocol.

        Returns the mean query loss with the meta-initialised parameters and with
        the adapted parameters, plus the exact delta.  At validation time
        ``support_frame`` is the last labelled TRAIN samples of this security and
        ``query_frame`` is the validation samples being scored.
        """
        query_loader = _loader_for(samples, query_frame, self.batch_size)
        _load_adaptable(self.model, self.meta_init)
        before = _query_loss(self.model, self.criterion, query_loader, self.device)
        adapted = self.adapt_for_prediction(samples, support_frame=support_frame,
                                            ticker=ticker)
        _load_adaptable(self.model, adapted)
        after = _query_loss(self.model, self.criterion, query_loader, self.device)
        _load_adaptable(self.model, self.meta_init)
        return {"ticker": ticker, "loss_before_adaptation": before,
                "loss_after_adaptation": after, "delta": after - before,
                "n_support": len(support_frame), "n_query": len(query_frame)}


def _loader_for(samples: SampleTable, frame: pd.DataFrame, batch_size: int):
    """Small deterministic loader over an explicit frame of sample rows."""
    from torch.utils.data import DataLoader

    dataset = V2SequenceDataset(samples, frame)
    generator = torch.Generator()
    generator.manual_seed(0)
    return DataLoader(dataset, batch_size=max(batch_size, 1), shuffle=False,
                      generator=generator)


def _parameter_distance(a: dict[str, torch.Tensor],
                        b: dict[str, torch.Tensor]) -> float:
    """Mean absolute distance between two adapted parameter sets."""
    total, n = 0.0, 0
    for name, tensor in a.items():
        other = b[name].to(tensor.device).float()
        total += float((tensor.float() - other).abs().mean())
        n += 1
    return total / max(n, 1)


def validation_support_frames(samples: SampleTable, *, ticker: str, n_support: int,
                              train_end: str, val_start: str) -> pd.DataFrame:
    """The last ``n_support`` labelled TRAIN samples of one security.

    These precede the validation window, so adapting on them uses no validation
    label.  The function refuses to return anything that is not strictly before
    the validation window.
    """
    frame = samples.frame
    subset = frame.loc[
        (frame["ticker"] == ticker)
        & (pd.to_datetime(frame["target_date"]) <= pd.Timestamp(train_end))
        & (pd.to_datetime(frame["target_date"]) < pd.Timestamp(val_start))
    ].sort_values("origin_date")
    if subset.empty:
        return subset.reset_index(drop=True)
    return subset.tail(n_support).reset_index(drop=True)


def meta_checkpoint_payload(model: ContextualLSTMTransformer,
                            meta_init: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in meta_init.items()}


def clone_state_dict(model: ContextualLSTMTransformer) -> dict[str, torch.Tensor]:
    return copy.deepcopy(model.state_dict())