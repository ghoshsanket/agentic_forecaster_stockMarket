#!/usr/bin/env bash
# =============================================================================
# MODEL V2 DEVELOPMENT PILOT -- unattended master run
#
# STAGED, NOT GRID.  There is no hyperparameter search here: the purpose is
# ARCHITECTURE CONTRIBUTION ANALYSIS with fixed settings.
#
#   0  environment (Research-local only, no sudo, no global packages)
#   1  ruff + pytest
#   2  build / verify the V2 context cache
#   3  synthetic learnable test      -> results/v2/sanity/synthetic_learnable.json
#   4  shuffled-label test          -> results/v2/sanity/shuffled_labels.json
#   5  V2-A on DEV A + DEV B   (2 shared-model fits)
#   6  V2-B on DEV A + DEV B   (2)
#   7  V2-C on DEV A + DEV B   (2)
#   8  CONTEXT_SIGNAL_GATE
#        gate FAILS -> summarise and STOP SUCCESSFULLY
#        gate PASSES -> V2-D (2), V2-E (2)
#   9  adaptation evaluation; V2-F only if adaptation shows a signal
#  10  select exactly ONE winner on DEV A+B, 3-seed stability (11/42/73)
#  11  freeze results/v2/v2_dev_selection.json
#  12  ONE 2021 lockbox run through scripts/run_v2_lockbox.py
#
# ABSOLUTE CONSTRAINTS (enforced in code, asserted per run)
#   * no target date >= 2022-01-01 is ever read or scored
#   * the development script can never score the 2021 lockbox
#   * FINAL_TEST is never set and run_recovered_paper.py is never invoked
#   * PAPER_REFERENCE is never imported; no published metric is a target
#   * the paper reconstruction, its results and its datasets are untouched
# =============================================================================

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ---- Research-local environment (idempotent) ------------------------------
# Locate the Research root by looking for its environment script, so the pilot
# behaves the same whether or not RESEARCH_ROOT was exported by the shell.
if [[ -z "${RESEARCH_ROOT:-}" ]]; then
  for candidate in "$(cd "$REPO_ROOT/.." && pwd)" "$(cd "$REPO_ROOT/../.." && pwd)"; do
    if [[ -f "$candidate/scripts/research-env.sh" ]]; then
      RESEARCH_ROOT="$candidate"
      break
    fi
  done
fi
export RESEARCH_ROOT="${RESEARCH_ROOT:-}"
if [[ -f "${RESEARCH_ROOT}/scripts/research-env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${RESEARCH_ROOT}/scripts/research-env.sh"
fi
if [[ -z "${RESEARCH_ROOT}" ]]; then
  echo "ERROR: could not locate the Research root (no scripts/research-env.sh found)" >&2
  exit 2
fi
export AGENTIC_PROJECT_ROOT="$REPO_ROOT"
export AGENTIC_OUTPUT_ROOT="${AGENTIC_OUTPUT_ROOT:-$RESEARCH_ROOT/output/agentic-forecaster}"
export AGENTIC_DATA_ROOT="${AGENTIC_DATA_ROOT:-$RESEARCH_ROOT/dataset}"
export AGENTIC_PROCESSED_DATA_ROOT="${AGENTIC_PROCESSED_DATA_ROOT:-$AGENTIC_DATA_ROOT/agentic-forecaster/processed}"
export UV_PROJECT_ENVIRONMENT="$AGENTIC_PROJECT_ROOT/.venv"
export PATH="$RESEARCH_ROOT/tools/bin:$AGENTIC_PROJECT_ROOT/.venv/bin:$PATH"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

DEVICE="${DEVICE:-auto}"
V2_LOCKBOX_SEED=42
CONFIG_DIR="configs/v2"
LOG_DIR="$AGENTIC_OUTPUT_ROOT/v2/logs"
LOCK="$LOG_DIR/v2_dev_pilot.lock"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/v2_dev_pilot.log"
: >"$LOG"

log() { printf '%s  %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "$LOG"; }

# ---- single-instance lock -------------------------------------------------
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "ERROR: a V2 development pilot is already in progress (lock: $LOCK)" >&2
  exit 75
fi
echo "$$" >&9

trap 'log "FAILED at line $LINENO"; exit 1' ERR

# The development path must NOT be able to score the lockbox year.
unset V2_LOCKBOX || true

log "=== MODEL V2 development pilot start ==="
log "config dir : $CONFIG_DIR"
log "device    : $DEVICE"
log "output    : $AGENTIC_OUTPUT_ROOT/v2"
log "processed : $AGENTIC_PROCESSED_DATA_ROOT/v2"

run_variant() {          # run_variant <VARIANT-LETTER> <FOLD> [SEED]
  local letter="$1" fold="$2" seed="${3:-42}"
  local config
  config="$(ls "$CONFIG_DIR"/v2_${letter}_*.yaml | head -1)"
  log "      $letter / $fold / seed $seed  ($config)"
  uv run --frozen python scripts/run_v2_experiment.py \
    --config "$config" --fold "$fold" --seed "$seed" --device "$DEVICE" \
    >>"$LOG" 2>&1
}

gate_value() {           # gate_value <jq-ish python expr over the summary>
  uv run --frozen python - "$1" <<'PYEOF' >>"$LOG" 2>&1
import json, sys
from pathlib import Path
summary = json.loads(Path("results/v2/v2_dev_summary.json").read_text())
print(eval(sys.argv[1]))          # trusted, fixed expressions from this script
PYEOF
}

# ---- 0. environment -------------------------------------------------------
log "[0/12] environment"
uv sync --all-extras --frozen >>"$LOG" 2>&1
uv run --frozen python scripts/verify_environment.py >>"$LOG" 2>&1
log "      environment verified"

# ---- 1. static analysis + tests ------------------------------------------
log "[1/12] ruff"
uv run --frozen python -m ruff check . >>"$LOG" 2>&1
log "[2/12] pytest"
uv run --frozen python -m pytest -p no:warnings -q >>"$LOG" 2>&1
log "      lint + tests green"

# ---- 2. V2 context cache ---------------------------------------------------
log "[3/12] build / verify the V2 context store"
uv run --frozen python scripts/build_v2_context.py >>"$LOG" 2>&1
uv run --frozen python scripts/build_v2_context.py --verify >>"$LOG" 2>&1
log "      context store verified"

# ---- 3/4. synthetic sanity gate -------------------------------------------
log "[4/12] synthetic learnable test"
uv run --frozen python scripts/run_v2_sanity.py --mode learnable >>"$LOG" 2>&1
log "[5/12] shuffled-label control"
uv run --frozen python scripts/run_v2_sanity.py --mode shuffled >>"$LOG" 2>&1
log "      sanity gate passed"

# ---- 5/6/7. first stage: V2-A, V2-B, V2-C on both dev folds ---------------
for letter in a b c; do
  for fold in V2_DEV_FOLD_A V2_DEV_FOLD_B; do
    log "[6/12] V2-$letter on $fold"
    run_variant "$letter" "$fold" 42
  done
done
log "      6 shared-model fits complete (3 variants x 2 folds)"

# ---- 8. the context signal gate -------------------------------------------
log "[7/12] evaluating CONTEXT_SIGNAL_GATE"
uv run --frozen python scripts/summarize_v2_dev.py >>"$LOG" 2>&1
GATE="$(gate_value "summary['context_signal_gate']['passed']")"
log "      context signal gate passed = $GATE"

if [[ "$GATE" != "True" ]]; then
  log "[STOP] CONTEXT_SIGNAL_GATE FAILED"
  log "       same-data context has NOT produced enough signal to justify"
  log "       multi-task / FiLM / meta-learning complexity."
  log "       V2-D, V2-E and V2-F are deliberately NOT run."
  log "       summary : results/v2/v2_dev_summary.json"
  log "       report  : results/v2/V2_DEV_REPORT.md"
  log "=== MODEL V2 development pilot complete (gate failed, stopped successfully) ==="
  exit 0
fi

# ---- 9. second stage: multi-task and FiLM ---------------------------------
for letter in d e; do
  for fold in V2_DEV_FOLD_A V2_DEV_FOLD_B; do
    log "[8/12] V2-$letter on $fold"
    run_variant "$letter" "$fold" 42
  done
done
uv run --frozen python scripts/summarize_v2_dev.py >>"$LOG" 2>&1

ADAPT="$(gate_value "summary['adaptation_evaluation']['V2-E_vs_V2-D'].get('materially_better') or summary['adaptation_evaluation']['V2-E_vs_V2-C'].get('materially_better')")"
log "      adaptation materially better = $ADAPT"

RUN_F="False"
if [[ "$ADAPT" == "True" ]]; then
  RUN_F="True"
  for fold in V2_DEV_FOLD_A V2_DEV_FOLD_B; do
    log "[9/12] V2-F on $fold (adaptation signal present)"
    run_variant f "$fold" 42
  done
  uv run --frozen python scripts/summarize_v2_dev.py >>"$LOG" 2>&1
else
  log "[9/12] V2-F SKIPPED: adaptation showed no material signal."
  log "       meta-learning is optional here and must NOT be assumed useful."
fi

# ---- 10. select ONE winner, then seed stability ---------------------------
SELECTED="$(gate_value "summary['selection']['selected']")"
log "[10/12] selected winner = $SELECTED (DEV A+B macro accuracy only)"
LETTER="$(printf '%s' "$SELECTED" | tr '[:upper:]' '[:lower:]' | sed 's/^v2-//')"

for seed in 11 73; do
  for fold in V2_DEV_FOLD_A V2_DEV_FOLD_B; do
    log "       seed stability: $SELECTED / $fold / seed $seed"
    run_variant "$LETTER" "$fold" "$seed"
  done
done
uv run --frozen python scripts/summarize_v2_dev.py >>"$LOG" 2>&1

STABILITY="$(gate_value "json.dumps(summary['seed_stability'].get('$SELECTED', {}), indent=2)")"
log "$STABILITY"

# ---- 11. freeze the selection --------------------------------------------
log "[11/12] freezing the development selection"
uv run --frozen python scripts/summarize_v2_dev.py \
  --freeze-selection "$SELECTED" >>"$LOG" 2>&1
log "      frozen: results/v2/v2_dev_selection.json"

# ---- 12. ONE 2021 lockbox run --------------------------------------------
log "[12/12] single 2021 lockbox run (seed $V2_LOCKBOX_SEED)"
V2_LOCKBOX=1 uv run --frozen python scripts/run_v2_lockbox.py \
  --config "$(ls "$CONFIG_DIR"/v2_${LETTER}_*.yaml | head -1)" \
  --variant "$SELECTED" --seed "$V2_LOCKBOX_SEED" --device "$DEVICE" \
  >>"$LOG" 2>&1
uv run --frozen python scripts/summarize_v2_dev.py --include-lockbox >>"$LOG" 2>&1

log "=== MODEL V2 development pilot complete ==="
log "selected  : $SELECTED"
log "summary   : results/v2/v2_dev_summary.json"
log "report    : results/v2/V2_DEV_REPORT.md"
log "selection : results/v2/v2_dev_selection.json"
log "ledger    : results/v2/experiment_ledger.csv"
log "2022/2023 : never touched"