"""PRE-COVID MULTI-HORIZON direction forecasting: objectives, trading-day targets,
track isolation and the 2019 lockbox firewall.

WHAT THIS TRACK IS
------------------
One question only:

    "Does the existing PRE-COVID price-derived dataset contain more predictable
     directional information at 3-, 5- or 10-trading-day horizons than at the
     one-day horizon?"

Only the FORECAST HORIZON changes.  No new data, no sentiment, no news, no
2020+, no architecture search and no meta-learning.  The feature set is the
existing V2 stationary stock schema and the data regime is the existing
PRE-COVID regime, so any difference in the result is attributable to the horizon
and to nothing else.

HORIZON MEANS TRADING OBSERVATIONS
----------------------------------
``H`` counts FUTURE TRADING OBSERVATIONS of that security, never calendar days::

    H = 1   ->  target row = t + 1 trading observation
    H = 3   ->  target row = t + 3 trading observations
    H = 5   ->  target row = t + 5 trading observations
    H = 10  ->  target row = t + 10 trading observations

A weekend or an exchange holiday is simply not an observation, so it is skipped
by construction.  ``date + timedelta(days=H)`` is NEVER used: that would silently
manufacture a target on a non-trading day and would not skip a holiday week.

TARGET DEFINITION (identical for every horizon)::

    future_log_return_H = log(Close[t+H] / Close[t])
    y_H                 = 1 if future_log_return_H > 0 else 0

``H = 1`` reproduces the existing PRE-COVID one-day target
(``y_direction``) exactly, which is asserted in code rather than assumed.

NO FUTURE PATH INFORMATION
--------------------------
For a 10-trading-day target the model still sees data only through ``t``.  The
intermediate closes ``t+1 .. t+H-1`` exist ONLY while deriving the label; they
are never features, never context and never scaler inputs.  The sample frame has
exactly one row per (security, origin), and the input sequence ends at the
origin row by construction (``V2SequenceDataset``).

2019 IS SEALED
--------------
Five chronological development folds score 2014-2018.  2019 is read exactly
once, by ``scripts/run_multi_horizon_lockbox.py``, and only with
``MULTI_HORIZON_LOCKBOX=1`` AND a frozen ``frozen_horizon_model.json``.
Development folds additionally refuse any target whose ``target_end_date``
falls in the lockbox year, so a horizon cannot be selected against 2019.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import features as feat
from .firewall import PRECOVID_FINAL_DATE, assert_pre_covid_dates
from .ledger import hash_payload

logger = logging.getLogger("agentic_forecaster.v2.horizons")

# ---------------------------------------------------------------------------
# regime identity
# ---------------------------------------------------------------------------

#: Track identity recorded in every config, manifest, ledger row and report.
TRACK_ID = "MULTI_HORIZON"
TRACK_LABEL = "PRE_COVID_MULTI_HORIZON_DIRECTION_TRACK"

#: Same regime as the PRE-COVID track: nothing from 2020 onward is ever consumed.
EXPERIMENT_REGIME = "PRE_COVID"
REGIME_LABEL = "PRE_COVID_EXPERIMENTAL_REGIME"
BIAS_LABEL = "SURVIVORSHIP_BIASED_FIXED_UNIVERSE_RESEARCH_TRACK"
UNIVERSE_PHRASE = "available securities from the reconstructed fixed universe"

#: Absolute final allowed TARGET date for the whole track, lockbox included.
FINAL_ALLOWED_DATE = str(PRECOVID_FINAL_DATE.date())

#: Environment switch that authorises the ONE 2019 lockbox run.  Deliberately
#: neither ``V2_LOCKBOX`` nor ``PRECOVID_LOCKBOX``: this track has its own switch.
LOCKBOX_ENV = "MULTI_HORIZON_LOCKBOX"
LOCKBOX_FOLD = "MH_LOCKBOX_2019"
LOCKBOX_YEAR = 2019

#: The lockbox year may not be read while horizons or models are being selected.
LOCKBOX_YEAR_START = pd.Timestamp(f"{LOCKBOX_YEAR}-01-01")
LOCKBOX_YEAR_END = pd.Timestamp(f"{LOCKBOX_YEAR}-12-31")

# ---------------------------------------------------------------------------
# horizons and objectives
# ---------------------------------------------------------------------------

#: ``1`` is the CONTROL horizon; ``3/5/10`` are the candidates under test.
HORIZONS: tuple[int, ...] = (1, 3, 5, 10)
CONTROL_HORIZON = 1
CANDIDATE_HORIZONS: tuple[int, ...] = (3, 5, 10)

#: Explicit objective identifiers, so no report can say "5D" without saying 5 DAYS.
OBJECTIVE_IDS: dict[int, str] = {
    1: "ABS_DIR_1D_CONTROL",
    3: "ABS_DIR_3D",
    5: "ABS_DIR_5D",
    10: "ABS_DIR_10D",
}

#: The exact target equation, recorded in the frozen manifest.
TARGET_EQUATION = "y_H = 1 if log(Close[t+H] / Close[t]) > 0 else 0"
RETURN_EQUATION = "future_log_return_H = log(Close[t+H] / Close[t])"
TRADING_DAY_RULE = (
    "H counts FUTURE TRADING OBSERVATIONS of the security: the target row is the "
    "H-th subsequent row of that security's ordered trading-observation sequence. "
    "Weekends and exchange holidays are not observations and are skipped by "
    "construction. date + timedelta(days=H) is never used."
)

#: Semantic wording required by the reporting rules.
def horizon_phrase(horizon: int) -> str:
    """Human wording that can never be mistaken for a one-day result."""
    if horizon == 1:
        return "1-trading-day directional accuracy"
    return f"{horizon}-trading-day directional accuracy"


def objective_id(horizon: int) -> str:
    try:
        return OBJECTIVE_IDS[int(horizon)]
    except KeyError as exc:
        raise ValueError(
            f"horizon {horizon!r} is not part of this experiment; "
            f"the tested horizons are {sorted(OBJECTIVE_IDS)}"
        ) from exc


#: Columns of the horizon target cache, in order.  ``close_t`` and
#: ``close_t_plus_h`` are provenance only; the model never sees them.
TARGET_COLUMNS: tuple[str, ...] = (
    "ticker",
    "origin_date",
    "target_end_date",
    "horizon",
    "future_log_return",
    "y",
    "close_t",
    "close_t_plus_h",
)

TARGET_CACHE_DIRNAME = "horizon_targets"

# ---------------------------------------------------------------------------
# track isolation
# ---------------------------------------------------------------------------

#: Results/ledger branch.  ``results/v2/`` and ``results/v2/pre_covid/`` are
#: never written by this track.
RESULTS_RELATIVE = Path("results/v2/multi_horizon")

TRACK_FILES: dict[str, str] = {
    "ledger": "experiment_ledger.csv",
    "universe_csv": "supervised_universe.csv",
    "targets_metadata": "target_schema.json",
    "freeze": "frozen_horizon_model.json",
    "report": "MULTI_HORIZON_REPORT.md",
    "summary": "multi_horizon_summary.json",
    "audit": "data_access_audit.json",
    "verification": "verification.json",
    "lockbox_report": "multi_horizon_lockbox_report.json",
}

#: Minimum columns of the append-only horizon ledger.
LEDGER_COLUMNS: tuple[str, ...] = (
    "experiment_id",
    "timestamp",
    "experiment_dir",
    "horizon",
    "objective_id",
    "model",
    "fold",
    "seed",
    "n_train",
    "n_validation",
    "n_tickers",
    "accuracy",
    "macro_accuracy",
    "balanced_accuracy",
    "f1",
    "roc_auc",
    "brier",
    "ece",
    "train_majority_baseline",
    "baseline_delta",
    "non_overlap_accuracy",
    "non_overlap_auc",
    "common_origin_accuracy",
    "common_origin_auc",
    "config_sha256",
    "target_schema_sha256",
    "2019_lockbox_evaluated",
    "post_2019_evaluated",
)

#: Models of the cheap first stage.  ``1D`` is a CONTROL, not a candidate.
SCREEN_MODELS: tuple[str, ...] = ("LOGISTIC", "HIST_GRADIENT_BOOSTING")
NEURAL_MODELS: tuple[str, ...] = ("SHARED_LSTM", "LSTM_TRANSFORMER")
NEURAL_LABELS: dict[str, str] = {
    "SHARED_LSTM": "N1 shared LSTM (unchanged V2-A architecture)",
    "LSTM_TRANSFORMER": "N2 shared LSTM + Transformer (unchanged V2-B architecture)",
}
SCREEN_LABELS: dict[str, str] = {
    "LOGISTIC": "M1 LogisticRegression",
    "HIST_GRADIENT_BOOSTING": "M2 HistGradientBoostingClassifier",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_track_config(path: str | Path) -> dict:
    """Load a track config, resolving its ``base:`` chain.

    A variant config declares only what it changes; the base supplies everything
    else, so the two can never drift apart.  The merge is a deep merge with the
    child winning, and the chain is resolved relative to each config's own
    directory.

    Mapping keys are normalised to strings so the payload hashes deterministically:
    a YAML block keyed by horizon (``3:``) would otherwise mix integer and string
    keys and break ``sort_keys=True``.
    """
    from agentic_forecaster.config import load_config

    def _load(current: Path, seen: list[Path]) -> dict:
        resolved = current.resolve()
        if resolved in seen:
            raise ValueError(f"circular base: chain at {resolved}")
        seen.append(resolved)
        payload = load_config(resolved)
        base = payload.pop("base", None)
        if base is None:
            return payload
        parent = _load(resolved.parent / str(base), seen)
        return _stringify_keys(_deep_merge(parent, payload))

    return _stringify_keys(_load(Path(path), []))


def _stringify_keys(payload):
    if isinstance(payload, dict):
        return {str(key): _stringify_keys(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_stringify_keys(item) for item in payload]
    return payload


def _deep_merge(parent: dict, child: dict) -> dict:
    merged = dict(parent)
    for key, value in child.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def results_root() -> Path:
    """``results/v2/multi_horizon``."""
    return repo_root() / RESULTS_RELATIVE


def processed_root() -> Path:
    """``$AGENTIC_PROCESSED_DATA_ROOT/v2/multi_horizon``."""
    from .store import v2_processed_root

    return v2_processed_root() / TRACK_ID.lower()


def runtime_root() -> Path:
    """``$AGENTIC_OUTPUT_ROOT/v2/multi_horizon``."""
    from .ledger import runtime_v2_root

    return runtime_v2_root(TRACK_ID.lower())


def target_cache_root() -> Path:
    return processed_root() / TARGET_CACHE_DIRNAME


@dataclass
class MultiHorizonTrack:
    """Resolved paths of the multi-horizon track."""

    results_root: Path = field(default_factory=results_root)
    processed_root: Path = field(default_factory=processed_root)
    runtime_root: Path = field(default_factory=runtime_root)

    def path(self, key: str) -> Path:
        return self.results_root / TRACK_FILES[key]

    @property
    def ledger(self) -> Path:
        return self.results_root / TRACK_FILES["ledger"]

    @property
    def target_cache(self) -> Path:
        return self.processed_root / TARGET_CACHE_DIRNAME


# ---------------------------------------------------------------------------
# the 2019 lockbox firewall
# ---------------------------------------------------------------------------

class MultiHorizonLockboxError(RuntimeError):
    """A multi-horizon development run attempted to read the 2019 lockbox."""

    __test__ = False


def lockbox_unlocked(explicit: bool | None = None) -> bool:
    """Resolve whether the ONE 2019 lockbox run is authorised.

    Precedence: explicit argument > ``MULTI_HORIZON_LOCKBOX`` > False.  The
    ordinary ``V2_LOCKBOX`` and ``PRECOVID_LOCKBOX`` switches have no effect here.
    """
    if explicit is not None:
        return bool(explicit)
    return os.environ.get(LOCKBOX_ENV, "") == "1"


def assert_no_lockbox_year_targets(values, *, where: str = "multi-horizon",
                                   unlocked: bool | None = None) -> str | None:
    """Reject any 2019 ``target_end_date`` unless the lockbox is unlocked.

    This is what keeps horizon and model selection off the lockbox year: a 10-day
    target that ends in January 2019 is already a lockbox read, so it is refused
    during development rather than quietly scored.
    """
    if lockbox_unlocked(unlocked):
        from .firewall import max_target_date

        return max_target_date(values)
    worst = None
    for value in values:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        ts = pd.Timestamp(value)
        if pd.isna(ts):
            continue
        worst = ts if worst is None or ts > worst else worst
        if LOCKBOX_YEAR_START <= ts <= LOCKBOX_YEAR_END:
            raise MultiHorizonLockboxError(
                f"{where}: refusing target_end_date {ts.date()}. {LOCKBOX_YEAR} is the "
                "multi-horizon LOCKBOX: it may be read exactly once, by "
                f"scripts/run_multi_horizon_lockbox.py with {LOCKBOX_ENV}=1, and only "
                "after horizon and model are frozen. During development no horizon "
                "may be selected using 2019."
            )
    return None if worst is None else worst.strftime("%Y-%m-%d")


def assert_horizon_boundary(target_end_dates, *, origin_dates=(), final_allowed_date:
                            str = FINAL_ALLOWED_DATE, where: str = "multi-horizon",
                            unlocked: bool | None = None) -> dict:
    """The two boundaries a multi-day target must respect.

    1. no target may end after the absolute final allowed date (2019-12-31), so a
       December-2019 origin whose 3/5/10-day target would fall in 2020 is DROPPED
       and no 2020 close is ever used to finish a target;
    2. no target may end inside the lockbox year during development.
    """
    assert_pre_covid_dates(origin_dates=origin_dates, target_dates=target_end_dates,
                           final_allowed_date=final_allowed_date, where=where)
    worst = assert_no_lockbox_year_targets(target_end_dates, where=where,
                                           unlocked=unlocked)
    return {"max_target_end_date": worst, "final_allowed_date": final_allowed_date}


def describe_firewall() -> dict:
    return {
        "track": TRACK_LABEL,
        "regime": REGIME_LABEL,
        "final_allowed_date": FINAL_ALLOWED_DATE,
        "first_forbidden_date": "2020-01-01",
        "post_covid_error": "PostCovidDataAccessError",
        "lockbox_env_var": LOCKBOX_ENV,
        "lockbox_fold": LOCKBOX_FOLD,
        "lockbox_year": LOCKBOX_YEAR,
        "lockbox_error": "MultiHorizonLockboxError",
        "trading_day_rule": TRADING_DAY_RULE,
        "target_equation": TARGET_EQUATION,
        "note": ("the raw source files still contain 2020-2025 bars; the PRE-COVID "
                 "source store is physically capped at 2019-12-31 and every horizon "
                 "target is capped again here, so no post-2019 close can reach a "
                 "label"),
    }


# ---------------------------------------------------------------------------
# horizon target construction
# ---------------------------------------------------------------------------

def build_horizon_target_frame(dataset_root: str | Path, *, variant: str = "adjusted",
                               tickers: list[str] | None = None,
                               horizons: tuple[int, ...] = HORIZONS,
                               final_allowed_date: str = FINAL_ALLOWED_DATE,
                               sheet_map: dict[str, str] | None = None,
                               where: str = "horizon targets") -> pd.DataFrame:
    """Build ``y_H`` for every horizon from the security's own trading rows.

    The target row is the H-th FUTURE TRADING OBSERVATION, taken from that
    security's ordered observation sequence, so weekends and exchange holidays are
    skipped by construction.  Rows whose target would end after
    ``final_allowed_date`` are dropped: no 2020 close ever finishes a target.
    """
    from .sectors import read_sheet_name_map

    root = Path(dataset_root)
    parquet_dir = root / variant / "parquet"
    boundary = pd.Timestamp(final_allowed_date)
    horizon_tuple = tuple(int(h) for h in horizons)
    if any(h < 1 for h in horizon_tuple):
        raise ValueError(f"horizons must be >= 1 trading observation, got {horizon_tuple}")
    sheet_map = sheet_map if sheet_map is not None else read_sheet_name_map(root)

    if tickers is None:
        tickers = sorted(p.stem for p in parquet_dir.glob("*.parquet"))

    blocks: list[pd.DataFrame] = []
    for ticker in sorted(tickers):
        path = parquet_dir / f"{sheet_map.get(ticker, ticker)}.parquet"
        if not path.is_file():
            logger.warning("no %s parquet for %s at %s", variant, ticker, path)
            continue
        bars = feat.normalise_ohlcv_frame(pd.read_parquet(path))
        bars = bars.loc[bars["date"] <= boundary]
        if len(bars) < max(horizon_tuple) + 1:
            continue
        dates = bars["date"].to_numpy(dtype="datetime64[ns]")
        closes = bars["close"].to_numpy(dtype=float)
        for horizon in horizon_tuple:
            # the H-th FUTURE observation: index i + H, not i + 1 with a date offset
            origin = dates[: len(dates) - horizon]
            target_end = dates[horizon:]
            close_t = closes[: len(closes) - horizon]
            close_future = closes[horizon:]
            future = np.log(close_future / close_t)
            blocks.append(pd.DataFrame({
                "ticker": str(ticker),
                "origin_date": origin,
                "target_end_date": target_end,
                "horizon": int(horizon),
                "future_log_return": future,
                "y": (close_future > close_t).astype(int),
                "close_t": close_t,
                "close_t_plus_h": close_future,
            }))

    if not blocks:
        raise ValueError(f"no horizon target rows could be built from {parquet_dir}")
    frame = pd.concat(blocks, ignore_index=True)
    # PHYSICAL CAP: a December-2019 origin whose 3/5/10-day target would land in
    # 2020 is dropped here, so no 2020 close is used to finish a 2019 target.
    before = len(frame)
    frame = frame.loc[frame["target_end_date"] <= boundary]
    dropped_post_boundary = before - len(frame)
    frame = frame.sort_values(["horizon", "ticker", "origin_date"]).reset_index(drop=True)
    frame = frame.loc[:, list(TARGET_COLUMNS)]

    assert_pre_covid_dates(origin_dates=frame["origin_date"],
                           target_dates=frame["target_end_date"],
                           final_allowed_date=final_allowed_date, where=where)
    logger.info("horizon targets: %d rows over horizons %s (%d dropped past %s)",
                len(frame), horizon_tuple, dropped_post_boundary, final_allowed_date)
    return frame


def assert_control_matches_store(targets: pd.DataFrame, store_targets: pd.DataFrame, *,
                                 horizon: int = CONTROL_HORIZON,
                                 where: str = "1D control") -> dict:
    """Prove the 1-day horizon target reproduces the existing ``y_direction``.

    The existing PRE-COVID one-day target is ``y_direction = int(close[t+1] >
    close[t])`` with ``raw_next_return = log(close[t+1]/close[t])``.  This compares
    both columns on the full intersection of ``(ticker, origin_date)`` keys and
    raises on ANY mismatch, so the control horizon cannot drift away from the
    experiment it is a control for.
    """
    control = targets.loc[targets["horizon"] == int(horizon)].copy()
    if control.empty:
        raise ValueError(f"no rows for the control horizon H={horizon}")
    existing = store_targets.loc[:, ["ticker", "origin_date", "target_date",
                                     "raw_next_return", "y_direction"]].copy()
    existing["origin_date"] = pd.to_datetime(existing["origin_date"])
    existing["target_date"] = pd.to_datetime(existing["target_date"])
    control["origin_date"] = pd.to_datetime(control["origin_date"])
    control["target_end_date"] = pd.to_datetime(control["target_end_date"])

    merged = control.merge(existing, on=["ticker", "origin_date"], how="inner",
                           validate="one_to_one")
    if merged.empty:
        raise ValueError("the control horizon shares no (ticker, origin_date) key "
                         "with the existing one-day target frame")
    label_mismatch = int((merged["y"].to_numpy(int)
                          != merged["y_direction"].to_numpy(int)).sum())
    date_mismatch = int((merged["target_end_date"].to_numpy(dtype="datetime64[ns]")
                         != merged["target_date"].to_numpy(dtype="datetime64[ns]")).sum())
    return_mismatch = merged.loc[
        ~np.isclose(merged["future_log_return"].to_numpy(float),
                    merged["raw_next_return"].to_numpy(float), rtol=0, atol=1e-12)]
    if label_mismatch or date_mismatch or len(return_mismatch):
        raise AssertionError(
            f"{where}: ABS_DIR_1D_CONTROL does not reproduce the existing one-day "
            f"target ({label_mismatch} label mismatches, {date_mismatch} target-date "
            f"mismatches, {len(return_mismatch)} return mismatches over "
            f"{len(merged)} shared keys). The control horizon must be identical to "
            "y_direction.")
    return {
        "objective_id": objective_id(horizon),
        "n_compared_keys": len(merged),
        "label_mismatches": 0,
        "target_date_mismatches": 0,
        "return_mismatches": 0,
        "identical": True,
        "compared_columns": ["y == y_direction", "target_end_date == target_date",
                             "future_log_return == raw_next_return"],
    }


def target_schema(targets: pd.DataFrame, *, store_sha256: str | None = None,
                  source_manifest_sha256: str | None = None,
                  feature_schema_sha256: str | None = None,
                  final_allowed_date: str = FINAL_ALLOWED_DATE,
                  horizons: tuple[int, ...] = HORIZONS) -> dict:
    """The machine-readable definition of every target in this experiment."""
    per_horizon = {}
    for horizon in horizons:
        block = targets.loc[targets["horizon"] == horizon]
        per_horizon[str(horizon)] = {
            "objective_id": objective_id(horizon),
            "horizon_trading_observations": int(horizon),
            "n_rows": len(block),
            "n_tickers": int(block["ticker"].nunique()) if len(block) else 0,
            "first_origin_date": (str(pd.Timestamp(block["origin_date"].min()).date())
                                  if len(block) else None),
            "last_origin_date": (str(pd.Timestamp(block["origin_date"].max()).date())
                                 if len(block) else None),
            "last_target_end_date": (str(pd.Timestamp(block["target_end_date"].max()).date())
                                     if len(block) else None),
            "positive_rate": float(block["y"].mean()) if len(block) else None,
        }
    payload = {
        "track": TRACK_ID,
        "horizons": [int(h) for h in horizons],
        "objectives": {str(h): objective_id(h) for h in horizons},
        "target_equation": TARGET_EQUATION,
        "return_equation": RETURN_EQUATION,
        "trading_day_rule": TRADING_DAY_RULE,
        "columns": list(TARGET_COLUMNS),
        "final_allowed_date": final_allowed_date,
        "post_2019_rows_possible": False,
        "store_sha256": store_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "feature_schema_sha256": feature_schema_sha256,
        "feature_schema": list(feat.STOCK_FEATURE_NAMES),
        "per_horizon": per_horizon,
    }
    payload["target_schema_sha256"] = hash_payload(payload)
    return payload


def write_target_cache(targets: pd.DataFrame, *, root: Path | None = None,
                       schema: dict | None = None,
                       force: bool = False) -> dict:
    """Persist the horizon target cache and its schema (idempotent)."""
    from agentic_forecaster.utils import atomic_json_dump, ensure_dir

    out = Path(root) if root is not None else target_cache_root()
    ensure_dir(out)
    parquet = out / "targets.parquet"
    metadata_path = out / "metadata.json"
    payload = schema or target_schema(targets)
    if parquet.is_file() and metadata_path.is_file() and not force:
        cached = json.loads(metadata_path.read_text())
        if cached.get("target_schema_sha256") == payload.get("target_schema_sha256"):
            logger.info("horizon target cache hit: %s", out)
            return cached
    targets.to_parquet(parquet, index=False)
    payload = dict(payload)
    payload["target_cache_path"] = str(parquet)
    payload["n_rows"] = len(targets)
    atomic_json_dump(payload, metadata_path)
    return payload


def load_target_cache(root: Path | None = None, *, final_allowed_date: str
                      = FINAL_ALLOWED_DATE) -> tuple[pd.DataFrame, dict]:
    """Load the horizon target cache, refusing anything past the boundary."""
    base = Path(root) if root is not None else target_cache_root()
    metadata_path = base / "metadata.json"
    parquet = base / "targets.parquet"
    if not (metadata_path.is_file() and parquet.is_file()):
        raise FileNotFoundError(
            f"No multi-horizon target cache at {base}. Run "
            "scripts/build_multi_horizon_targets.py first.")
    metadata = json.loads(metadata_path.read_text())
    frame = pd.read_parquet(parquet)
    frame["origin_date"] = pd.to_datetime(frame["origin_date"])
    frame["target_end_date"] = pd.to_datetime(frame["target_end_date"])
    assert_pre_covid_dates(origin_dates=frame["origin_date"],
                           target_dates=frame["target_end_date"],
                           final_allowed_date=final_allowed_date,
                           where="multi-horizon target cache load")
    return frame, metadata


# ---------------------------------------------------------------------------
# common-origin and non-overlapping sample selection
# ---------------------------------------------------------------------------

def horizon_target_keys(targets: pd.DataFrame, horizons: tuple[int, ...] = HORIZONS
                        ) -> pd.MultiIndex:
    """Origins whose target is available for EVERY listed horizon."""
    blocks = []
    for horizon in horizons:
        block = targets.loc[targets["horizon"] == horizon, ["ticker", "origin_date"]]
        blocks.append(block.assign(_h=int(horizon)))
    stacked = pd.concat(blocks, ignore_index=True)
    counts = stacked.groupby(["ticker", "origin_date"], sort=True)["_h"].nunique()
    complete = counts.loc[counts == len({int(h) for h in horizons})]
    return pd.MultiIndex.from_frame(complete.reset_index()[["ticker", "origin_date"]])


def common_origin_mask(frame: pd.DataFrame, keys: pd.MultiIndex) -> np.ndarray:
    """Boolean mask of ``frame`` rows whose origin is in ``keys``."""
    if len(frame) == 0:
        return np.zeros(0, dtype=bool)
    index = pd.MultiIndex.from_frame(frame.loc[:, ["ticker", "origin_date"]])
    return index.isin(keys)


def non_overlap_mask(frame: pd.DataFrame, *, horizon: int,
                     date_col: str = "origin_date") -> np.ndarray:
    """Keep every ``horizon``-th valid origin, per security, from a fixed start.

    Adjacent multi-day targets overlap in their future holding period, so the
    ALL-ORIGINS metrics treat dependent observations as independent.  This mask
    produces the sensitivity view: within each security the origins are ordered by
    date and positions ``0, H, 2H, ...`` are retained, which guarantees a spacing
    of at least ``H`` trading observations.  The starting offset is FIXED (the
    first valid origin in the split) and is never chosen from results.
    """
    if len(frame) == 0:
        return np.zeros(0, dtype=bool)
    keep = np.zeros(len(frame), dtype=bool)
    for index in frame.groupby("ticker", sort=True).groups.values():
        ordered = frame.loc[index].sort_values(date_col).index.to_numpy()
        positions = np.arange(len(ordered))
        keep[ordered[(positions % int(horizon)) == 0]] = True
    return keep


def non_overlap_spacing(frame: pd.DataFrame, *, horizon: int) -> dict:
    """Audit the realised spacing of a NON-OVERLAPPING selection."""
    gaps: list[int] = []
    ordered_dates = pd.to_datetime(frame["origin_date"])
    for _, block in frame.assign(_d=ordered_dates).groupby("ticker", sort=True):
        rows = block.sort_values("_d")
        if len(rows) < 2:
            continue
        gaps.extend(int(g) for g in rows["_d"].diff().dt.days.dropna().tolist())
    return {
        "horizon": int(horizon),
        "n_origins": len(frame),
        "n_tickers": int(frame["ticker"].nunique()) if len(frame) else 0,
        "min_calendar_gap_days": int(min(gaps)) if gaps else None,
        "note": ("calendar gaps are a lower bound on the trading-observation spacing; "
                 "the guarantee itself is positional (every H-th observation)"),
    }


# ---------------------------------------------------------------------------
# append-only ledger
# ---------------------------------------------------------------------------

def ledger_path(root: Path | None = None) -> Path:
    """The multi-horizon ledger.  Never the ordinary V2 or PRE-COVID ledger."""
    return (Path(root) / TRACK_FILES["ledger"]) if root is not None else results_root() / TRACK_FILES["ledger"]


def new_experiment_id(prefix: str = "MH") -> str:
    from .ledger import new_experiment_id as _v2_id

    return _v2_id(prefix)


def read_ledger(path: Path | None = None) -> list[dict]:
    target = Path(path) if path is not None else ledger_path()
    if not target.is_file():
        return []
    with target.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def append_row(record: dict, path: Path | None = None, *,
               lockbox_evaluated: bool = False) -> dict:
    """Append one immutable multi-horizon row.

    The helper REFUSES a development row that claims the 2019 lockbox was scored,
    so a leak cannot be recorded as if it were legitimate.
    """
    target = Path(path) if path is not None else ledger_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    row = {c: record.get(c, "") for c in LEDGER_COLUMNS}
    row["experiment_id"] = row["experiment_id"] or new_experiment_id()
    row["timestamp"] = row["timestamp"] or datetime.now(UTC).isoformat()
    row["horizon"] = int(row["horizon"])
    row["objective_id"] = row["objective_id"] or objective_id(row["horizon"])
    row["model"] = str(row["model"]).upper()

    if str(row["post_2019_evaluated"]).strip().lower() in ("true", "1", "yes"):
        raise ValueError(
            "Refusing to append a multi-horizon row claiming post_2019_evaluated=true. "
            "Nothing from 2020 onward may be consumed by any script of this track.")
    row["post_2019_evaluated"] = "false"

    claimed = str(row["2019_lockbox_evaluated"]).strip().lower() in ("true", "1", "yes")
    if claimed and not lockbox_evaluated:
        raise ValueError(
            "Refusing to append a development row claiming 2019_lockbox_evaluated=true. "
            f"Only scripts/run_multi_horizon_lockbox.py with {LOCKBOX_ENV}=1 may score 2019.")
    row["2019_lockbox_evaluated"] = "true" if lockbox_evaluated else "false"

    new_file = not target.exists()
    with target.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(LEDGER_COLUMNS))
        if new_file:
            writer.writeheader()
        writer.writerow({c: row[c] for c in LEDGER_COLUMNS})
    logger.info("multi-horizon ledger row appended: %s %s %s %s",
                row["objective_id"], row["model"], row["fold"], row["experiment_id"])
    return row


def track_manifest(extra: dict | None = None) -> dict:
    payload = {
        "track": TRACK_ID,
        "track_label": TRACK_LABEL,
        "experiment_regime": EXPERIMENT_REGIME,
        "regime_label": REGIME_LABEL,
        "survivorship_bias_label": BIAS_LABEL,
        "universe_phrase": UNIVERSE_PHRASE,
        "final_allowed_date": FINAL_ALLOWED_DATE,
        "horizons": list(HORIZONS),
        "objectives": {str(h): objective_id(h) for h in HORIZONS},
        "control_horizon": CONTROL_HORIZON,
        "candidate_horizons": list(CANDIDATE_HORIZONS),
        "screen_models": list(SCREEN_MODELS),
        "neural_models": list(NEURAL_MODELS),
        "results_root": str(results_root()),
        "runtime_root": str(runtime_root()),
        "processed_root": str(processed_root()),
        "firewall": describe_firewall(),
        "test_2022_2023_evaluated": False,
    }
    payload.update(extra or {})
    return payload