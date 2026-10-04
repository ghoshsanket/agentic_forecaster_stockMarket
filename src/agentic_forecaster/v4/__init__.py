"""V4 -- PRE-COVID HISTORICAL NEWS SENTIMENT + EVENT INFORMATION programme.

THE QUESTION
------------
    "Does point-in-time historical NEWS information -- headline sentiment, news
    intensity, and broad India event pressure -- add genuine directional signal
    beyond price features plus genuine Indian-market state?"

This is an INFORMATION-SIGNAL experiment.  V3 established that real Indian
market/volatility data adds a small but real ranking increment (best family
``X1_INDIA_MARKET`` under Logistic Regression, 3D/5D/10D), that global risk
information HURTS, and that the combined exogenous family dilutes the Indian
signal.  V4 therefore changes the information set again and measures the
increment over a ``B0_INDIA_MARKET_BASE`` control.  No neural complexity, no
meta-learning, and no architecture search.

THE PREDICTION TIMESTAMP (unchanged from every earlier track)
-------------------------------------------------------------
    AFTER the NSE market close on day ``t``.

News has a publication DATE but no reliable publication TIME, so the
conservative rule is absolute and is enforced in code, not by convention::

    only stories with  publish_date < t  may enter features at origin t

No story dated ``t`` may ever be used, weekend news is valid on the following
trading day, and ``indexed_date`` is provenance only -- never a substitute for
publication time.

SOURCE STRATEGY (post-pivot)
-----------------------------
The originally specified GDELT 1.0 GKG primary source was RETIRED before this
track ran (``data.gdeltproject.org/gdeltv1/`` returns HTTP 404 for every path,
including its own master file list), and Google BigQuery is unavailable in this
environment.  GDELT 2.x exists only as 15-minute archives, which the source
policy forbids bulk-downloading.  The priority order is therefore:

1. ``MEDIA_CLOUD_ONLINE_NEWS_ARCHIVE``   primary company-news source
2. ``MEDIA_CLOUD_WAYBACK_TITLE_BACKFILL`` only where audited, never merged blindly
3. ``EVENT_REGISTRY``                     optional, credential-gated, never assumed
4. ``GDELT2_EVENTS``                      secondary, broad India event pressure only
5. ``COMMON_CRAWL``                       not implemented; last-resort backfill only

``GDELT2_EVENTS`` is deliberately events-only: company sentiment must come from
Media Cloud, never from the GKG stream.

REGIME
------
Strictly PRE-COVID.  The absolute final allowed date is ``2019-12-31`` for stock
features, stock targets, news publication dates and event dates.  2019 is sealed
behind ``V4_SENTIMENT_LOCKBOX=1``.  The development folds are NOT hard-coded:
they are generated after the source coverage probe (``news_coverage_selected_folds``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

#: Track identity.
TRACK_ID = "V4_SENTIMENT_EVENT_PRECOVID"
TRACK_LABEL = "V4_PRE_COVID_NEWS_SENTIMENT_EVENT_TRACK"
EXPERIMENT_REGIME = "PRE_COVID"
REGIME_LABEL = "PRE_COVID_EXPERIMENTAL_REGIME"
BIAS_LABEL = "SURVIVORSHIP_BIASED_FIXED_UNIVERSE_RESEARCH_TRACK"
UNIVERSE_PHRASE = "available securities from the reconstructed fixed universe"

#: Absolute final allowed date for EVERY V4 date: features, targets, news
#: publication dates and event dates.
FINAL_ALLOWED_DATE = "2019-12-31"

#: The one-shot 2019 lockbox switch, distinct from V2/PRE-COVID/V3.
LOCKBOX_ENV = "V4_SENTIMENT_LOCKBOX"
LOCKBOX_YEAR = 2019

#: Feature families.  ``B0`` is the CONTROL every increment is measured on; the
#: combined family is last on purpose, because V3 showed combinations dilute.
FEATURE_FAMILIES: tuple[str, ...] = (
    "B0_INDIA_MARKET_BASE",
    "B1_HEADLINE_COMPANY_SENTIMENT",
    "B2_MARKET_NEWS_SENTIMENT",
    "B3_COMPANY_PLUS_MARKET_SENTIMENT",
    "B4_GDELT_EVENT_PRESSURE",
    "B5_ALL_NEWS_EVENT",
)
CONTROL_FAMILY = "B0_INDIA_MARKET_BASE"
INCREMENTAL_FAMILIES: tuple[str, ...] = FEATURE_FAMILIES[1:]

#: Target horizons, in trading observations.  3D/5D/10D carry the interesting
#: baseline behaviour; 1D is retained as the control.
HORIZONS: tuple[int, ...] = (1, 3, 5, 10)

#: News aggregation windows, in CALENDAR days (news exists seven days a week).
#: Every window is ``[t - window, t)`` and therefore excludes the origin day.
NEWS_WINDOWS: tuple[int, ...] = (1, 3, 7, 14, 30)

#: Event-pressure windows (GDELT2 starts 2015-02-18, so these exist for fewer
#: origins than the company-news windows; the shorter common period is reported).
EVENT_WINDOWS: tuple[int, ...] = (1, 3, 7, 14)

#: "Recent company news" is DEFINED HERE, before any evaluation, and is never
#: retuned afterwards: at least one TITLE_EXPLICIT article in the prior
#: ``NEWS_PRESENT_WINDOW`` calendar days.
NEWS_PRESENT_WINDOW = 3

#: Alias safety classes.  Only EXACT_SAFE and HISTORICAL_SAFE ever match
#: automatically; AMBIGUOUS and REJECTED never do.
ALIAS_EXACT_SAFE = "EXACT_SAFE"
ALIAS_HISTORICAL_SAFE = "HISTORICAL_SAFE"
ALIAS_AMBIGUOUS = "AMBIGUOUS"
ALIAS_REJECTED = "REJECTED"
AUTO_MATCH_ALIAS_CLASSES: tuple[str, ...] = (ALIAS_EXACT_SAFE, ALIAS_HISTORICAL_SAFE)

#: Title-relevance classes (Media Cloud searches indexed body text but returns
#: titles only, so a body-only match must not be scored as company sentiment).
TITLE_EXPLICIT = "TITLE_EXPLICIT"
BODY_MATCH_ONLY = "BODY_MATCH_ONLY"

#: Source-priority verdicts.  ``NewsAPI`` was probed and rejected for
#: insufficient historical depth on ordinary plans.
SOURCE_PRIORITY: tuple[str, ...] = (
    "MEDIA_CLOUD_ONLINE_NEWS_ARCHIVE",
    "MEDIA_CLOUD_WAYBACK_TITLE_BACKFILL",
    "EVENT_REGISTRY",
    "GDELT2_EVENTS",
    "COMMON_CRAWL",
)

#: First GDELT2 archive date; before this, event-pressure features are
#: unavailable and MUST NOT be filled.
GDELT2_FIRST_AVAILABLE = "2015-02-18"

#: Feasibility verdicts.
FEASIBILITY_VERDICTS: tuple[str, ...] = ("V4_NEWS_SOURCE_READY",
                                         "V4_NEWS_SOURCE_INSUFFICIENT")

#: Final signal vocabulary.
SIGNALS: tuple[str, ...] = ("NONE", "WEAK", "PROMISING", "STRONG")

#: The permitted recommendations.  Exactly one is reported, never executed.
NEXT_ACTIONS: tuple[str, ...] = (
    "USE_SENTIMENT_LSTM",
    "BUILD_RANKING_OBJECTIVE_WITH_SENTIMENT",
    "BUILD_FULLTEXT_FINANCIAL_SENTIMENT_DATASET",
    "PRICE_DIRECTION_65_PERCENT_NOT_SUPPORTED",
)

# ---------------------------------------------------------------------------
# isolated locations -- never inside results/v2 or results/v3
# ---------------------------------------------------------------------------

RESULTS_RELATIVE = Path("results/v4/pre_covid_sentiment")

TRACK_FILES: dict[str, str] = {
    "ledger": "experiment_ledger.csv",
    "report": "V4_SENTIMENT_REPORT.md",
    "summary": "v4_sentiment_summary.json",
    "mediacloud_probe": "MEDIACLOUD_PROBE.md",
    "mediacloud_probe_csv": "mediacloud_probe.csv",
    "gdelt_event_probe": "GDELT_EVENT_PROBE.md",
    "gdelt_event_probe_csv": "gdelt_event_probe.csv",
    "gdelt_event_manifest": "gdelt_event_source_manifest.csv",
    "alias_audit": "ALIAS_AUDIT.md",
    "alias_audit_csv": "alias_audit.csv",
    "alias_registry": "company_aliases_resolved.json",
    "source_completeness": "SOURCE_COMPLETENESS.md",
    "selected_folds": "news_coverage_selected_folds.json",
    "availability_audit": "AVAILABILITY_AUDIT.md",
    "data_access_audit": "data_access_audit.json",
    "freeze": "frozen_sentiment_model.json",
    "seed_stability": "seed_stability.json",
    "verification": "verification.json",
    "lockbox_report": "v4_sentiment_lockbox_report.json",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def results_root() -> Path:
    """``results/v4/pre_covid_sentiment`` -- never inside ``results/v2``/``v3``."""
    return repo_root() / RESULTS_RELATIVE


def data_root() -> Path:
    """``$AGENTIC_DATA_ROOT/news/pre_covid`` (raw provider snapshots)."""
    from agentic_forecaster.config import get_env_roots

    return Path(get_env_roots()["AGENTIC_DATA_ROOT"]) / "news" / "pre_covid"


def runtime_root() -> Path:
    """``$AGENTIC_OUTPUT_ROOT/v4/pre_covid_sentiment``."""
    from agentic_forecaster.config import get_env_roots

    return Path(get_env_roots()["AGENTIC_OUTPUT_ROOT"]) / "v4" / "pre_covid_sentiment"


def processed_root() -> Path:
    """``$AGENTIC_PROCESSED_DATA_ROOT/v4/pre_covid_sentiment``."""
    from agentic_forecaster.config import get_env_roots

    return (Path(get_env_roots()["AGENTIC_PROCESSED_DATA_ROOT"]) / "v4"
            / "pre_covid_sentiment")


def raw_root(family: str | None = None) -> Path:
    base = data_root() / "raw"
    return base / family if family else base


def manifests_root() -> Path:
    return data_root() / "manifests"


def research_root() -> Path:
    """``$RESEARCH_ROOT``, resolved exactly as ``agentic_forecaster.config`` does."""
    import os

    return Path(os.environ.get("RESEARCH_ROOT", str(repo_root().parent.parent)))


def tmp_root() -> Path:
    """Research-local temporary storage for streamed archives.

    Temporary archives are written here and deleted after a successful
    reduction.  ``/tmp`` is avoided on purpose: a streamed reduction must not
    depend on a small system partition.
    """
    return research_root() / "tmp"


def secrets_root() -> Path:
    """``$RESEARCH_ROOT/secrets`` -- the ONLY place a credential may live."""
    return research_root() / "secrets"


@dataclass
class V4Track:
    """Resolved paths of the V4 sentiment/event track."""

    results_root: Path = field(default_factory=results_root)
    processed_root: Path = field(default_factory=processed_root)
    runtime_root: Path = field(default_factory=runtime_root)

    def path(self, key: str) -> Path:
        return self.results_root / TRACK_FILES[key]