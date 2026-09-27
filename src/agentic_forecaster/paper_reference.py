"""PAPER REFERENCE metrics transcribed from the publication.

These values are **not** produced by this codebase.  They are the published
numbers, kept in exactly one place so that every comparison table joins
against the same source.

DOI: 10.1109/IEMENTECH202669403.2026.11434302
"""

from __future__ import annotations

PAPER_REFERENCE: dict[str, dict[str, float]] = {
    "attention_lstm_calibrated": {
        "accuracy": 0.815,
        "f1": 0.60,
        "brier": 0.205,
        "precision_at_3_up": 0.70,
        "precision_at_3_down": 0.65,
    },
    "attention_lstm_raw": {
        "accuracy": 0.815,
        "f1": 0.59,
        "brier": 0.240,
        "precision_at_3_up": 0.70,
        "precision_at_3_down": 0.65,
    },
    "plain_lstm": {
        "accuracy": 0.690,
        "f1": 0.57,
        "brier": 0.230,
        "precision_at_3_up": 0.62,
        "precision_at_3_down": 0.60,
    },
    "random_forest": {
        "accuracy": 0.772,
        "f1": 0.55,
        "brier": 0.245,
        "precision_at_3_up": 0.58,
        "precision_at_3_down": 0.52,
    },
}

PROVENANCE = "paper_reference"
DESCRIPTION = (
    "Metrics transcribed from the published paper. PAPER REFERENCE — NOT "
    "produced by this code."
)


def as_dict() -> dict:
    return {
        "provenance": PROVENANCE,
        "description": DESCRIPTION,
        "paper": {
            "title": "Explanation-First Agentic Forecaster for Stock Market",
            "doi": "10.1109/IEMENTECH202669403.2026.11434302",
        },
        "models": PAPER_REFERENCE,
    }
