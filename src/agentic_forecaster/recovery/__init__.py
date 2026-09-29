"""Performance-recovery harness.

Goal: recover the publication's predictive performance *scientifically*, without
repeatedly tuning against the final 2022/2023 test years.

The central safety mechanism is :mod:`agentic_forecaster.recovery.firewall`: in
``search_mode`` any attempt to score a date on or after ``TEST_FIREWALL_START``
raises :class:`TestSetFirewallError`.  Model search may therefore only ever see
pre-2022 validation data.
"""

from agentic_forecaster.recovery.firewall import (
    TEST_FIREWALL_END,
    TEST_FIREWALL_START,
    TestSetFirewallError,
    assert_config_windows_safe,
    assert_pre_test_dates,
    date_in_firewalled_window,
    describe_firewall,
    firewall_guard,
    is_search_mode,
)
from agentic_forecaster.recovery.folds import (
    PAPER_FOLDS,
    SEARCH_FOLDS,
    SearchFold,
    fold_config_for,
)
from agentic_forecaster.recovery.ledger import (
    LEDGER_COLUMNS,
    append_experiment,
    read_ledger,
)
from agentic_forecaster.recovery.variants import (
    ARCHITECTURES,
    CALIBRATION_METHODS,
    FEATURE_FAMILIES,
    RSI_METHODS,
    SCALERS,
    TRAINING_LENGTHS,
    WEIGHT_DECAYS,
    build_feature_indicators,
)

__all__ = [
    "ARCHITECTURES",
    "CALIBRATION_METHODS",
    "FEATURE_FAMILIES",
    "LEDGER_COLUMNS",
    "PAPER_FOLDS",
    "RSI_METHODS",
    "SCALERS",
    "SEARCH_FOLDS",
    "TEST_FIREWALL_END",
    "TEST_FIREWALL_START",
    "TRAINING_LENGTHS",
    "WEIGHT_DECAYS",
    "SearchFold",
    "TestSetFirewallError",
    "append_experiment",
    "assert_config_windows_safe",
    "assert_pre_test_dates",
    "build_feature_indicators",
    "date_in_firewalled_window",
    "describe_firewall",
    "firewall_guard",
    "fold_config_for",
    "is_search_mode",
    "read_ledger",
]
