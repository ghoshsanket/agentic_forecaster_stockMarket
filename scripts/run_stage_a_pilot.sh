#!/usr/bin/env bash
# Stage-A PILOT: a small, controlled, PRE-2022 comparison.
#
# 3 candidate families x 3 search folds x 8 representative stocks = 9 experiment
# runs / 72 model fits. Every fit scores a PRE-2022 validation window only.
#
# This is deliberately NOT the full Stage-A grid (which would multiply the
# training-length axis across every ticker and fold). The pilot exists to answer
# three questions before an expensive search is worth running:
#
#   P0 vs P1  -> was the 3-epoch cap too restrictive?
#   P1 vs P2  -> do the paper-evidence features (SMA/Bollinger/OBV) help?
#   P1/P2     -> is there any pre-2022 predictive signal at all?
#
# HARD CONSTRAINTS
#   * no date >= 2022-01-01 is ever scored (the firewall enforces this)
#   * calibration is NONE, so validation metrics are from RAW p(up); Stage D
#     compares calibration methods separately
#   * volume_mode is raw, the Phase-1 reference; log1p is a Stage-B option
#   * run_recovered_paper.py is never invoked and FINAL_TEST is never set
#
# Usage:  bash scripts/run_stage_a_pilot.sh

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
log "tickers: $TICKERS"
log "folds  : ${FOLDS[*]}"

# 3 configurations x 3 folds = 9 experiment runs (72 model fits)
for fold in "${FOLDS[@]}"; do
  run_one "P0_T3_F1"     T3   F1 "$fold"   # original 3-epoch control
  run_one "P1_T100_F1"  T100 F1 "$fold"   # proper early stopping
  run_one "P2_T100_F2"  T100 F2 "$fold"   # + paper-evidence features
done

log "=== Stage-A pilot complete: 9 experiment runs / 72 model fits ==="
log "Next: uv run --frozen python scripts/summarize_stage_a_pilot.py"
