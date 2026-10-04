"""The PRE-COVID EXPERIMENTAL REGIME: track definition and supervised-universe
eligibility.

WHAT THIS TRACK IS
------------------
A forecasting model trained AND evaluated entirely before 2020.  The complete
usable horizon ends at ``2019-12-31``; nothing from 2020 onward may be consumed
for feature construction, market/sector context, cross-sectional ranks, scaler
fitting, training, validation, early stopping, architecture selection,
meta-learning episodes, confidence selection or metrics.

WHAT THIS IS NOT
----------------
This is **not** a claim that COVID caused any earlier failure.  It establishes
one thing only: whether a model built and evaluated entirely in a pre-COVID
regime behaves differently.  The 2020+ period is a separate, future
regime-aware / event-aware problem.

UNIVERSE CAVEAT, STATED NOT HIDDEN
----------------------------------
The universe is a fixed, later-reconstructed constituent list projected
backwards.  Every PRE-COVID result therefore carries the label
``SURVIVORSHIP_BIASED_FIXED_UNIVERSE_RESEARCH_TRACK``.  Reports say "available
securities from the reconstructed fixed universe", never "the NIFTY-50
constituents on that historical date".  This limitation does not invalidate the
architecture experiment, but it must travel with every number.

TWO POPULATIONS, KEPT SEPARATE
-------------------------------
SUPERVISED population
    only securities that pass every eligibility rule below; these are the only
    ones that contribute supervised gradients and that are scored.
CONTEXT population
    every security of the reconstructed fixed universe that actually has a valid
    bar on the historical date ``t``.  A later-listed security starts
    contributing once it genuinely exists.  Nothing is ever backfilled.

ELIGIBILITY (all must hold; no date after 2019-12-31 is consulted)
------------------------------------------------------------------
A. sufficient TRAIN history by 2016-12-31 (the first fold's train end);
B. >= ``min_train_samples`` usable supervised TRAIN samples in PRECOVID_DEV_A;
C. >= ``min_year_samples`` usable samples in calendar 2017;
D. >= ``min_year_samples`` usable samples in calendar 2018;
E. >= ``min_year_samples`` usable samples in calendar 2019;
F. every required stationary feature can be generated causally (implied by the
   finite-window requirement, checked explicitly).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .firewall import PRECOVID_FINAL_DATE, assert_pre_covid_dates
from .ledger import hash_payload

logger = logging.getLogger("agentic_forecaster.v2.precovid")

#: Regime identity recorded in every config, manifest, ledger row and report.
EXPERIMENT_REGIME = "PRE_COVID"
REGIME_LABEL = "PRE_COVID_EXPERIMENTAL_REGIME"
BIAS_LABEL = "SURVIVORSHIP_BIASED_FIXED_UNIVERSE_RESEARCH_TRACK"
UNIVERSE_PHRASE = "available securities from the reconstructed fixed universe"

#: Fold names.  Selection may see 2017 and 2018; 2019 is the ONE lockbox.
DEV_FOLD_A = "PRECOVID_DEV_A"
DEV_FOLD_B = "PRECOVID_DEV_B"
LOCKBOX_FOLD = "PRECOVID_LOCKBOX"
DEV_FOLDS: tuple[str, ...] = (DEV_FOLD_A, DEV_FOLD_B)

#: Eligibility thresholds (fixed before the universe was derived).
MIN_TRAIN_SAMPLES = 1000
MIN_YEAR_SAMPLES = 180

#: Columns of the frozen universe record, in order.
UNIVERSE_COLUMNS: tuple[str, ...] = (
    "ticker",
    "raw_first_date",
    "first_usable_sample",
    "train_samples_through_2016",
    "samples_2017",
    "samples_2018",
    "samples_2019",
    "eligible",
    "exclusion_reason",
    "sector",
)

TRACK_FILES: dict[str, str] = {
    "ledger": "experiment_ledger.csv",
    "universe_csv": "pre_covid_supervised_universe.csv",
    "universe_yaml": "supervised_universe.yaml",
    "selection": "pre_covid_dev_selection.json",
    "report": "PRE_COVID_V2_REPORT.md",
    "summary": "pre_covid_v2_summary.json",
    "audit": "data_access_audit.json",
    "logistic": "precovid_logistic_baseline.json",
}


def repo_results_root() -> Path:
    return Path(__file__).resolve().parents[3] / "results" / "v2" / "pre_covid"


def processed_root() -> Path:
    """``$AGENTIC_PROCESSED_DATA_ROOT/v2/pre_covid``."""
    from .store import precovid_processed_root

    return precovid_processed_root()


def runtime_root() -> Path:
    """``$AGENTIC_OUTPUT_ROOT/v2/pre_covid``."""
    from .ledger import runtime_v2_root

    return runtime_v2_root("pre_covid")


def assert_regime_date(values, *, where: str = "precovid") -> None:
    """Reject any 2020+ date in PRE-COVID code, at the point of access."""
    assert_pre_covid_dates(feature_dates=values, origin_dates=values, target_dates=values,
                           where=where)


@dataclass
class EligibilityRules:
    """The fixed eligibility thresholds."""

    min_train_samples: int = MIN_TRAIN_SAMPLES
    min_year_samples: int = MIN_YEAR_SAMPLES
    train_end: str = "2016-12-31"       # PRECOVID_DEV_A train end
    dev_year_a: str = "2017"
    dev_year_b: str = "2018"
    lockbox_year: str = "2019"
    final_allowed_date: str = str(PRECOVID_FINAL_DATE.date())

    def to_dict(self) -> dict:
        return {
            "min_train_samples": self.min_train_samples,
            "min_year_samples": self.min_year_samples,
            "train_end": self.train_end,
            "dev_year_a": self.dev_year_a,
            "dev_year_b": self.dev_year_b,
            "lockbox_year": self.lockbox_year,
            "final_allowed_date": self.final_allowed_date,
            "criteria": [
                "A. sufficient TRAIN history by the first fold's train end",
                f"B. >= {self.min_train_samples} usable TRAIN samples in PRECOVID_DEV_A",
                f"C. >= {self.min_year_samples} usable samples in {self.dev_year_a}",
                f"D. >= {self.min_year_samples} usable samples in {self.dev_year_b}",
                f"E. >= {self.min_year_samples} usable samples in {self.lockbox_year}",
                "F. all required stationary features generated causally (finite window)",
            ],
        }


def evaluate_eligibility(samples: pd.DataFrame, *, ticker: str, raw_first_date,
                         sector: str, rules: EligibilityRules | None = None
                         ) -> dict:
    """Evaluate the eligibility rules for ONE security.

    ``samples`` is the security's own supervised sample table (origin/target
    dates).  No date after the PRE-COVID boundary is consulted: the frame is
    checked first, so a leak cannot influence the count.
    """
    rules = rules or EligibilityRules()
    assert_pre_covid_dates(origin_dates=samples["origin_date"],
                           target_dates=samples["target_date"],
                           final_allowed_date=rules.final_allowed_date,
                           where=f"eligibility {ticker}")
    origin = pd.to_datetime(samples["origin_date"])
    target = pd.to_datetime(samples["target_date"])
    features_finite = bool(samples.attrs.get("features_finite", True))

    train = int(((origin <= pd.Timestamp(rules.train_end))
                 & (target <= pd.Timestamp(rules.train_end))).sum())
    per_year = {
        year: int((origin.dt.year == int(year)).sum())
        for year in (rules.dev_year_a, rules.dev_year_b, rules.lockbox_year)
    }

    reasons: list[str] = []
    if not features_finite:
        reasons.append("F: non-finite stationary features")
    if train < rules.min_train_samples:
        reasons.append(
            f"A/B: only {train} usable TRAIN samples through {rules.train_end} "
            f"(need {rules.min_train_samples})")
    for label, year in (("C", rules.dev_year_a), ("D", rules.dev_year_b),
                        ("E", rules.lockbox_year)):
        if per_year[year] < rules.min_year_samples:
            reasons.append(f"{label}: only {per_year[year]} samples in {year} "
                           f"(need {rules.min_year_samples})")

    eligible = not reasons
    return {
        "ticker": ticker,
        "raw_first_date": None if raw_first_date is None else str(
            pd.Timestamp(raw_first_date).date()),
        "first_usable_sample": (str(pd.Timestamp(origin.min()).date())
                                if len(origin) else None),
        "train_samples_through_2016": train,
        "samples_2017": per_year[rules.dev_year_a],
        "samples_2018": per_year[rules.dev_year_b],
        "samples_2019": per_year[rules.lockbox_year],
        "eligible": bool(eligible),
        "exclusion_reason": "; ".join(reasons) if reasons else "",
        "sector": sector,
    }


def derive_supervised_universe(samples: pd.DataFrame, *, raw_first_dates: dict,
                               sector_of: dict, candidates: list[str] | None = None,
                               rules: EligibilityRules | None = None,
                               where: str = "eligibility") -> pd.DataFrame:
    """Derive ONE FIXED PRE-COVID supervised universe, deterministically.

    ``samples`` must already be restricted to the PRE-COVID store.  ``candidates``
    is every security of the reconstructed fixed universe that has any usable
    history, so a candidate that produced NO usable sample at all still appears in
    the record with an explicit reason instead of silently disappearing.

    The output is sorted by ticker, so the frozen list is reproducible, and every
    excluded security carries a reason.
    """
    rules = rules or EligibilityRules()
    assert_pre_covid_dates(origin_dates=samples["origin_date"],
                           target_dates=samples["target_date"],
                           final_allowed_date=rules.final_allowed_date, where=where)
    pool = sorted(candidates) if candidates else sorted(samples["ticker"].unique())
    rows: list[dict] = []
    for ticker in pool:
        group = samples.loc[samples["ticker"] == ticker]
        if group.empty:
            rows.append({
                "ticker": str(ticker),
                "raw_first_date": (None if raw_first_dates.get(str(ticker)) is None
                                   else str(pd.Timestamp(
                                       raw_first_dates[str(ticker)]).date())),
                "first_usable_sample": None,
                "train_samples_through_2016": 0,
                "samples_2017": 0,
                "samples_2018": 0,
                "samples_2019": 0,
                "eligible": False,
                "exclusion_reason": (
                    "A-F: no usable supervised sample at all (insufficient causal "
                    "history, or no sector peer for the leave-one-out sector "
                    "context)"),
                "sector": sector_of.get(str(ticker), "UNKNOWN"),
            })
            continue
        rows.append(evaluate_eligibility(
            group, ticker=str(ticker), raw_first_date=raw_first_dates.get(str(ticker)),
            sector=sector_of.get(str(ticker), "UNKNOWN"), rules=rules))
    frame = pd.DataFrame(rows, columns=list(UNIVERSE_COLUMNS))
    return frame.sort_values("ticker").reset_index(drop=True)


def eligible_tickers(universe: pd.DataFrame) -> list[str]:
    return sorted(universe.loc[universe["eligible"], "ticker"].tolist())


def universe_hash(universe: pd.DataFrame) -> str:
    """Stable hash of the frozen eligible list (the universe fingerprint).

    Built from the canonical CSV TEXT so that hashing the frozen file and
    recomputing it from the file on disk agree exactly.
    """
    payload = (universe.loc[:, list(UNIVERSE_COLUMNS)].to_csv(index=False)
               + json.dumps({"tickers": eligible_tickers(universe),
                             "rules": EligibilityRules().to_dict(),
                             "regime": EXPERIMENT_REGIME, "bias": BIAS_LABEL},
                            sort_keys=True))
    return hash_payload(payload)


def write_universe(universe: pd.DataFrame, *, csv_path: Path, yaml_path: Path,
                   store_sha256: str | None = None) -> dict:
    """Freeze the derived universe to CSV and YAML BEFORE any training runs."""
    import yaml

    from agentic_forecaster.utils import atomic_json_dump

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    universe.to_csv(csv_path, index=False)
    rules = EligibilityRules()
    payload = {
        "experiment_regime": EXPERIMENT_REGIME,
        "regime_label": REGIME_LABEL,
        "survivorship_bias_label": BIAS_LABEL,
        "universe_phrase": UNIVERSE_PHRASE,
        "final_allowed_date": rules.final_allowed_date,
        "n_candidates": len(universe),
        "n_eligible": len(eligible_tickers(universe)),
        "eligible_tickers": eligible_tickers(universe),
        "excluded": {
            str(r["ticker"]): r["exclusion_reason"]
            for r in universe.loc[~universe["eligible"]].to_dict("records")
        },
        "rules": rules.to_dict(),
        "store_sha256": store_sha256,
        "universe_sha256": universe_hash(universe),
        "context_population": (
            "every security of the reconstructed fixed universe with a valid bar on "
            "date t; later listings contribute from the day they genuinely exist and "
            "are never backfilled"),
        "frozen": True,
        "note": ("frozen BEFORE any PRE-COVID variant was trained; the count is "
                 "derived, never guessed"),
    }
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False, width=100),
                         encoding="utf-8")
    del atomic_json_dump
    return payload


def load_universe(yaml_path: Path) -> dict:
    import yaml

    return yaml.safe_load(Path(yaml_path).read_text())


@dataclass
class PreCovidTrack:
    """Resolved paths of the PRE-COVID track."""

    results_root: Path = field(default_factory=repo_results_root)
    processed_root: Path = field(default_factory=processed_root)
    runtime_root: Path = field(default_factory=runtime_root)

    def path(self, key: str) -> Path:
        return self.results_root / TRACK_FILES[key]

    @property
    def ledger(self) -> Path:
        return self.results_root / TRACK_FILES["ledger"]


def assert_no_post_covid_rows(frame: pd.DataFrame, *columns: str,
                              where: str = "precovid frame") -> dict:
    """Raise if any supplied date column carries a post-2019 row."""
    present = [c for c in columns if c in frame.columns]
    assert_pre_covid_dates(**{f"{c}_dates": frame[c] for c in present},
                           where=where)
    return {
        "checked_columns": present,
        "n_rows": len(frame),
        "max_date": (None if not present or frame.empty
                     else str(max(pd.Timestamp(frame[c].max()).date()
                                  for c in present))),
    }


def guard(value, *, where: str = "precovid"):
    """Small helper for scripts: reject a single 2020+ date."""
    if value is None:
        return None
    assert_pre_covid_dates(feature_dates=[value], where=where)
    return value


def cohort_summary(universe: pd.DataFrame) -> dict:
    """Counts used in the report: eligible vs excluded, by sector."""
    eligible = universe.loc[universe["eligible"]]
    return {
        "n_candidates": len(universe),
        "n_eligible": len(eligible),
        "n_excluded": int((~universe["eligible"]).sum()),
        "eligible_tickers": eligible_tickers(universe),
        "excluded_tickers": sorted(universe.loc[~universe["eligible"], "ticker"]),
        "eligible_by_sector": {
            str(k): int(v) for k, v in eligible.groupby("sector").size().items()},
        "lockbox_year_usable_by_every_eligible": bool(
            (eligible["samples_2019"] > 0).all()) if len(eligible) else False,
        "survivorship_bias_label": BIAS_LABEL,
    }


def nan_free(matrix: np.ndarray) -> bool:
    return bool(np.isfinite(matrix).all())