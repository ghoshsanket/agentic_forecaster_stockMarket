#!/usr/bin/env bash
# Stage-A PILOT: a small, controlled, PRE-2022 comparison.
#
# The epoch budget is AUTHOR-CONFIRMED at 10, so it is NOT a search axis. The
# pilot runs in two generations over the same 8 stocks and the same 3 folds:
#
#   --generation author-confirmed   PRIMARY
#       T10 x {F1,F2}                     2 x 3 x 8 = 48 fits
#       Asks: at the confirmed schedule, does the compact F1 feature set or the
#       expanded F2 set (SMA5/SMA20/SMA5-SMA20, Bollinger bands and %B, OBV)
#       discriminate on pre-2022 validation? F1 and F2 are held at EXACTLY 10
#       epochs so any difference is attributable to features alone.
#
#   --generation diagnostic         RETAINED, NOT candidates
#       T3 x F1, T100 x {F1,F2}            3 x 3 x 8 = 72 fits
#       Measures what the confirmed 10-epoch budget costs. T3 was a
#       reconstruction shortcut and T100 was a recovery diagnostic; neither was
#       the author's schedule and a diagnostic win does not license selecting it.
#
#   --generation all                both
#
# HARD CONSTRAINTS
#   * no date >= 2022-01-01 is ever scored (the firewall enforces this)
#   * calibration is NONE, so validation metrics are from RAW p(up); calibration
#     is recovered separately once the model configuration is selected
#   * volume_mode is raw, the Phase-1 reference; log1p is a Stage-B option
#   * run_recovered_paper.py is never invoked and FINAL_TEST is never set
#
# Usage:  bash scripts/run_stage_a_pilot.sh [--generation author-confirmed|diagnostic|all]


set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ---- Research-local environment (idempotent) ------------------------------
if [[ -f "${RESEARCH_ROOT:-}/scripts/research-env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${RESEARCH_ROOT}/scripts/research-env.sh"
fi
export RESEARCH_ROOT="${RESEARCH_ROOT:-$(cd "$REPO_ROOT/.." && pwd)}"
export AGENTIC_PROJECT_ROOT="$REPO_ROOT"
export AGENTIC_MODEL_ROOT="${AGENTIC_MODEL_ROOT:-$RESEARCH_ROOT/models/agentic-forecaster}"
export AGENTIC_OUTPUT_ROOT="${AGENTIC_OUTPUT_ROOT:-$RESEARCH_ROOT/output/agentic-forecaster}"
export AGENTIC_DATA_ROOT="${AGENTIC_DATA_ROOT:-$RESEARCH_ROOT/dataset}"
export UV_PROJECT_ENVIRONMENT="$AGENTIC_PROJECT_ROOT/.venv"
export PATH="$RESEARCH_ROOT/tools/bin:$AGENTIC_PROJECT_ROOT/.venv/bin:$PATH"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

CONFIG="configs/reproduction_search/paper_snapshot_2025_unadjusted_perf.yaml"
TICKERS="RELIANCE,TCS,INFY,HDFCBANK,ITC,LT,SUNPHARMA,TATASTEEL"
FOLDS=(SEARCH_FOLD_A SEARCH_FOLD_B SEARCH_FOLD_C)
DEVICE="${DEVICE:-auto}"

LOG_DIR="$AGENTIC_OUTPUT_ROOT/reproduction_recovery/logs"
LOCK="$LOG_DIR/stage_a_pilot.lock"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/stage_a_pilot.log"

# ---- single-instance lock -------------------------------------------------
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "ERROR: another Stage-A pilot is already running (lock: $LOCK)" >&2
  exit 75
fi
echo "$$" >&9

# ---- helpers --------------------------------------------------------------
log() { printf '%s  %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "$LOG"; }

# ---------------------------------------------------------------------------
# Generation selection
#
#   author-confirmed  PRIMARY  : T10 x {F1,F2}  = 2 x 3 x 8 = 48 fits
#   diagnostic        RETAINED : T3 x F1, T100 x {F1,F2} = 3 x 3 x 8 = 72 fits
#   all               both generations
#
# The epoch budget is AUTHOR-CONFIRMED at 10, so it is not a search axis. T3 and
# T100 are retained ONLY to measure what the confirmed budget costs; neither is
# a reproduction candidate and a diagnostic win does not license selecting it.
# ---------------------------------------------------------------------------
GENERATION="author-confirmed"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --generation) GENERATION="$2"; shift 2 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
case "$GENERATION" in
  author-confirmed|diagnostic|all) ;;
  *) echo "unknown generation: $GENERATION" >&2; exit 2 ;;
esac


# run_one <label> <training_length> <feature_family>
run_one() {
  local label="$1" tl="$2" fam="$3" fold="$4"
  log "START ${label} / ${fold}"
  # Explicit arguments only. No custom training logic lives in this script.
  uv run --frozen python scripts/run_reproduction_search.py \
    --config "$CONFIG" \
    --stage A \
    --search-fold "$fold" \
    --tickers "$TICKERS" \
    --training-length "$tl" \
    --feature-family "$fam" \
    --rsi-method R1_wilder \
    --lookback 30 \
    --scaler standard \
    --volume-mode raw \
    --architecture A0 \
    --dropout 0.2 \
    --weight-decay 0.0001 \
    --class-weighting none \
    --calibration none \
    --seed 42 \
    --device "$DEVICE" >>"$LOG" 2>&1
  log "DONE  ${label} / ${fold}"
}

trap 'log "FAILED at line $LINENO"; exit 1' ERR

log "=== Stage-A pilot start (pre-2022 only) ==="
log "generation: $GENERATION"
log "tickers: $TICKERS"
log "folds  : ${FOLDS[*]}"

# ---- generation 2: the author-confirmed 10-epoch pilot (PRIMARY) -----------
# F1 and F2 are compared at EXACTLY 10 epochs so the difference is attributable
# to the feature set alone and cannot be confounded by training length.
if [[ "$GENERATION" == "author-confirmed" || "$GENERATION" == "all" ]]; then
  for fold in "${FOLDS[@]}"; do
    run_one "P10_T10_F1" T10_AUTHOR_CONFIRMED F1 "$fold"
    run_one "P10_T10_F2" T10_AUTHOR_CONFIRMED F2 "$fold"
  done
  log "author-confirmed generation complete: 6 experiment runs / 48 model fits"
fi

# ---- generation 1: epoch diagnostics (retained, NOT candidates) ------------
if [[ "$GENERATION" == "diagnostic" || "$GENERATION" == "all" ]]; then
  for fold in "${FOLDS[@]}"; do
    run_one "P0_T3_F1"        T3_RECONSTRUCTION_SHORTCUT F1 "$fold"
    run_one "P1_T100_F1"      T100_DIAGNOSTIC          F1 "$fold"
    run_one "P2_T100_F2"      T100_DIAGNOSTIC          F2 "$fold"
  done
  log "diagnostic generation complete: 9 experiment runs / 72 model fits"
fi

log "=== Stage-A pilot complete ==="
log "Next: uv run --frozen python scripts/summarize_stage_a_pilot.py"

