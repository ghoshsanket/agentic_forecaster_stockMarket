"""V3 -- PRE-COVID EXOGENOUS MARKET-INFORMATION programme.

THE QUESTION
------------
    "Does point-in-time market, volatility, global-risk, currency and commodity
     information add genuine directional signal beyond the stock's own
     price-derived features?"

The V2 and PRE-COVID programmes established that the 27 stationary stock-derived
features carry at most a weak one-day edge, and the multi-horizon study showed the
same at 3, 5 and 10 trading days.  V3 therefore does NOT search architectures
again.  It changes the INFORMATION SET and measures the increment.

THE PREDICTION TIMESTAMP (unchanged from every earlier track)
-------------------------------------------------------------
    AFTER the NSE market close on day t.

The system predicts the stock's future movement after ``t`` for 1D (control), 3D,
5D and 10D.  An exogenous value may enter the feature set at origin ``t`` only if
it was genuinely available by that timestamp.  A foreign market close that happens
hours after the NSE close is therefore NOT available at ``t``, no matter that the
provider labels it with the same calendar date.

THE FIVE FEATURE FAMILIES
--------------------------
===================== =====================================================
``X0_STOCK_ONLY``     the existing 27 stationary stock features (CONTROL)
``X1_INDIA_MARKET``   X0 + real NIFTY 50 / NIFTY Bank / India VIX + verified
                       sector-index features
``X2_GLOBAL_RISK``    X0 + conservatively lagged global equity and volatility
``X3_MACRO_COMMODITY``X0 + USD/INR, crude, gold, rates, dollar index
``X4_ALL_EXOGENOUS``  X0 + every accepted exogenous family
===================== =====================================================

No Cartesian proliferation of feature combinations is created: these four
incremental families are the whole experiment.

REGIME
------
Strictly PRE-COVID.  The absolute final allowed date is ``2019-12-31`` for stock
features, stock targets AND exogenous observations.  Development uses 2014-2018
only.  2019 is sealed behind ``V3_PRECOVID_LOCKBOX=1``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

#: Track identity.
TRACK_ID = "V3_EXOGENOUS_PRECOVID"
TRACK_LABEL = "V3_PRE_COVID_EXOGENOUS_MARKET_INFORMATION_TRACK"
EXPERIMENT_REGIME = "PRE_COVID"
REGIME_LABEL = "PRE_COVID_EXPERIMENTAL_REGIME"
BIAS_LABEL = "SURVIVORSHIP_BIASED_FIXED_UNIVERSE_RESEARCH_TRACK"
UNIVERSE_PHRASE = "available securities from the reconstructed fixed universe"

#: Absolute final allowed date for EVERY V3 date: features, exogenous
#: observations and targets.
FINAL_ALLOWED_DATE = "2019-12-31"

#: The one-shot 2019 lockbox switch.  Deliberately distinct from ``V2_LOCKBOX``
#: and ``PRECOVID_LOCKBOX``.
LOCKBOX_ENV = "V3_PRECOVID_LOCKBOX"
LOCKBOX_FOLD = "MH_LOCKBOX_2019"
LOCKBOX_YEAR = 2019

#: Feature families, X0 first: X0 is the CONTROL every increment is measured on.
FEATURE_FAMILIES: tuple[str, ...] = (
    "X0_STOCK_ONLY", "X1_INDIA_MARKET", "X2_GLOBAL_RISK",
    "X3_MACRO_COMMODITY", "X4_ALL_EXOGENOUS",
)
CONTROL_FAMILY = "X0_STOCK_ONLY"
EXOGENOUS_FAMILIES: tuple[str, ...] = FEATURE_FAMILIES[1:]

#: The three permitted recommendations.  Exactly one is reported, never executed.
NEXT_ACTIONS: tuple[str, ...] = (
    "USE_EXOGENOUS_LSTM",
    "ADD_HISTORICAL_SENTIMENT_AND_EVENTS",
    "BUILD_RELATIVE_RANKING_MODEL",
)

#: Final signal vocabulary.
SIGNALS: tuple[str, ...] = ("NONE", "WEAK", "PROMISING", "STRONG")

# ---------------------------------------------------------------------------
# isolated locations
# ---------------------------------------------------------------------------

RESULTS_RELATIVE = Path("results/v3/pre_covid_exogenous")

TRACK_FILES: dict[str, str] = {
    "ledger": "experiment_ledger.csv",
    "report": "V3_EXOGENOUS_REPORT.md",
    "summary": "v3_exogenous_summary.json",
    "source_audit": "SOURCE_AUDIT.md",
    "source_audit_json": "source_audit.json",
    "availability_audit": "AVAILABILITY_AUDIT.md",
    "availability_audit_json": "availability_audit.json",
    "data_access_audit": "data_access_audit.json",
    "registry": "source_registry_resolved.json",
    "freeze": "frozen_exogenous_model.json",
    "seed_stability": "seed_stability.json",
    "verification": "verification.json",
    "lockbox_report": "v3_lockbox_report.json",
}

#: Columns of the append-only V3 ledger.
LEDGER_COLUMNS: tuple[str, ...] = (
    "experiment_id",
    "timestamp",
    "experiment_dir",
    "model",
    "feature_family",
    "horizon",
    "objective_id",
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
    "stock_only_auc",
    "incremental_auc",
    "common_sample_auc",
    "non_overlap_auc",
    "config_sha256",
    "source_manifest_sha256",
    "feature_schema_sha256",
    "2019_lockbox_evaluated",
    "post_2019_evaluated",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def results_root() -> Path:
    """``results/v3/pre_covid_exogenous`` -- never inside ``results/v2``."""
    return repo_root() / RESULTS_RELATIVE


def data_root() -> Path:
    """``$AGENTIC_DATA_ROOT/exogenous/pre_covid`` (raw provider snapshots)."""
    from agentic_forecaster.config import get_env_roots

    return Path(get_env_roots()["AGENTIC_DATA_ROOT"]) / "exogenous" / "pre_covid"


def runtime_root() -> Path:
    """``$AGENTIC_OUTPUT_ROOT/v3/pre_covid_exogenous``."""
    from agentic_forecaster.config import get_env_roots

    return Path(get_env_roots()["AGENTIC_OUTPUT_ROOT"]) / "v3" / "pre_covid_exogenous"


def processed_root() -> Path:
    """``$AGENTIC_PROCESSED_DATA_ROOT/v3/pre_covid_exogenous``."""
    from agentic_forecaster.config import get_env_roots

    return (Path(get_env_roots()["AGENTIC_PROCESSED_DATA_ROOT"]) / "v3"
            / "pre_covid_exogenous")


def raw_root(family: str | None = None) -> Path:
    base = data_root() / "raw"
    return base / family if family else base


def manifests_root() -> Path:
    return data_root() / "manifests"


@dataclass
class V3Track:
    """Resolved paths of the V3 exogenous track."""

    results_root: Path = field(default_factory=results_root)
    processed_root: Path = field(default_factory=processed_root)
    runtime_root: Path = field(default_factory=runtime_root)

    def path(self, key: str) -> Path:
        return self.results_root / TRACK_FILES[key]

    @property
    def ledger(self) -> Path:
        return self.results_root / TRACK_FILES["ledger"]

    @property
    def store(self) -> Path:
        return self.processed_root / "exogenous_store"
