#!/usr/bin/env bash
# =============================================================================
# V3 -- PRE-COVID EXOGENOUS MARKET-INFORMATION PROGRAMME (master run)
#
# Answers one question:
#   "Does point-in-time market, volatility, global-risk, currency and commodity
#    information add genuine directional signal beyond the stock's own
#    price-derived features?"
#
# This is a DATA-INFORMATION experiment.  The decision metric is incremental AUC over
# the stock-only control X0 on a COMMON sample.
#
# STAGED, WITH HARD STOPS
#    1-5   environment, lint, tests
#    6-7   probe every declared external source -> SOURCE_AUDIT
#    8-9   download only the accepted candidates, hash them
#   10-11  build the causally aligned exogenous store -> AVAILABILITY_AUDIT
#   12     ABORT on any causality violation
#   13-14  X0..X4 Logistic screen, then HistGradientBoosting screen
#          (4 horizons x 5 folds x 2 models x 5 families = 200 cheap fits)
#   15     common-sample incremental metrics
#   16     FEATURE-FAMILY GATE
#     -> if NO family passes: report, STOP SUCCESSFULLY, train NO neural model,
#        do NOT open 2019
#   17     select at most TOP 2 family+horizon candidates
#   18     SHARED_LSTM_EXOGENOUS on those candidates
#   19     NEURAL GATE -> if nothing qualifies: report, STOP
#   20     LSTM_TRANSFORMER_EXOGENOUS ablation only
#   21     choose one model
#   22     seed stability -> UNSTABLE stops the programme
#   23-24  freeze the exogenous model, verify the frozen hashes
#   25     ONE 2019 lockbox run with V3_PRECOVID_LOCKBOX=1
#   26     final report
#
# NEVER accesses 2020+.  2019 is read only by step 25.
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ---- Research-local environment (idempotent) ------------------------------
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
if [[ -z "$RESEARCH_ROOT" ]]; then
  echo "ERROR: could not locate the Research root" >&2
  exit 2
fi
export AGENTIC_PROJECT_ROOT="$REPO_ROOT"
export AGENTIC_OUTPUT_ROOT="${AGENTIC_OUTPUT_ROOT:-$RESEARCH_ROOT/output/agentic-forecaster}"
export AGENTIC_DATA_ROOT="${AGENTIC_DATA_ROOT:-$RESEARCH_ROOT/dataset}"
export AGENTIC_PROCESSED_DATA_ROOT="${AGENTIC_PROCESSED_DATA_ROOT:-$AGENTIC_DATA_ROOT/agentic-forecaster/processed}"
export UV_PROJECT_ENVIRONMENT="$AGENTIC_PROJECT_ROOT/.venv"
export PATH="$RESEARCH_ROOT/tools/bin:$AGENTIC_PROJECT_ROOT/.venv/bin:$PATH"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# The earlier tracks' lockbox authorisations must never be present here: V3 has its
# own one-shot 2019 switch.
unset V2_LOCKBOX PRECOVID_LOCKBOX MULTI_HORIZON_LOCKBOX || true

CONFIGS="configs/v3"
RESULTS="results/v3/pre_covid_exogenous"
RUNTIME="${AGENTIC_OUTPUT_ROOT}/v3/pre_covid_exogenous"
LOG_DIR="$RUNTIME/logs"
LOCK="$LOG_DIR/v3_precovid_program.lock"
DEVICE="${V3_DEVICE:-auto}"
SEEDS="${V3_SEEDS:-11 42 73}"
mkdir -p "$LOG_DIR" "$RESULTS"
LOG="$LOG_DIR/v3_precovid_program.log"
: >"$LOG"

log() { printf '%s  %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "$LOG"; }

# ---- single-instance lock -------------------------------------------------
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "FATAL: another V3 exogenous programme run holds $LOCK" >&2
  exit 1
fi

summary_value() {          # summary_value <python expression over `summary`>
  uv run --frozen python - "$1" <<'PY' 2>>"$LOG"
import json, sys
from pathlib import Path
path = Path("results/v3/pre_covid_exogenous/v3_exogenous_summary.json")
if not path.is_file():
    print(""); raise SystemExit(0)
summary = json.loads(path.read_text())
print(eval(sys.argv[1], {"summary": summary, "json": json}))
PY
}

summarize() {
  local extra=()
  [[ -f "$RESULTS/verification.json" ]] && extra+=(--verification "$RESULTS/verification.json")
  uv run --frozen python scripts/summarize_v3_exogenous.py \
    --config "$CONFIGS/precovid_exogenous_base.yaml" "${extra[@]}" "$@" >>"$LOG" 2>&1
}

log "=== V3 PRE-COVID EXOGENOUS programme start ==="
log "repo        : $REPO_ROOT"
log "configs     : $CONFIGS"
log "results     : $RESULTS"
log "runtime     : $RUNTIME"
log "device      : $DEVICE"
log "regime      : PRE_COVID (final allowed target date 2019-12-31)"
log "prediction  : AFTER the NSE close on day t"
log "families    : X0_STOCK_ONLY (control) / X1_INDIA / X2_GLOBAL / X3_MACRO / X4_ALL"
log "lockbox     : 2019, V3_PRECOVID_LOCKBOX=1, one run only"

# ---- 1. environment --------------------------------------------------------
log "[1/26] environment"
uv sync --all-extras --frozen >>"$LOG" 2>&1
uv run --frozen python scripts/verify_environment.py >>"$LOG" 2>&1
log "      environment verified"

# ---- 2. lint ---------------------------------------------------------------
log "[2/26] ruff"
set +e
uv run --frozen python -m ruff check . >>"$LOG" 2>&1
RUFF_STATUS=$?
set -e
log "      ruff exit=$RUFF_STATUS"
if [[ "$RUFF_STATUS" -ne 0 ]]; then
  log "FATAL: ruff failed; the programme must not continue"
  exit 1
fi

# ---- 3. tests --------------------------------------------------------------
log "[3/26] pytest"
set +e
uv run --frozen python -m pytest -p no:warnings -q >>"$LOG" 2>&1
PYTEST_STATUS=$?
set -e
log "      pytest exit=$PYTEST_STATUS"
uv run --frozen python - "$RESULTS/verification.json" "$RUFF_STATUS" "$PYTEST_STATUS" <<'PY' >>"$LOG" 2>&1
import datetime, json, sys
from pathlib import Path
path, ruff_status, pytest_status = sys.argv[1:4]
Path(path).write_text(json.dumps({
    "track": "V3_EXOGENOUS_PRECOVID",
    "recorded_at": datetime.datetime.now(datetime.UTC).isoformat(),
    "ruff": "pass" if ruff_status == "0" else f"fail (exit {ruff_status})",
    "pytest": "pass" if pytest_status == "0" else f"fail (exit {pytest_status})",
    "pytest_exit_code": int(pytest_status),
    "pytest_command": "uv run --frozen python -m pytest -p no:warnings -q",
    "ruff_command": "uv run --frozen python -m ruff check .",
}, indent=2, sort_keys=True) + "\n")
PY
if [[ "$PYTEST_STATUS" -ne 0 ]]; then
  log "FATAL: pytest failed; the programme must not continue"
  exit 1
fi

# ---- 4. probe every declared source ---------------------------------------
log "[4/26] probe every declared external source"
uv run --frozen python scripts/probe_v3_sources.py --no-store >>"$LOG" 2>&1
log "      SOURCE_AUDIT written"

# ---- 5. download the accepted candidates ----------------------------------
log "[5/26] download the ACCEPTED source candidates and hash them"
uv run --frozen python scripts/download_v3_exogenous.py >>"$LOG" 2>&1
grep -E "^\[download\]" "$LOG" | tail -20 >>"$LOG" || true
log "      accepted sources downloaded"

# ---- 6. build the causally aligned store ----------------------------------
log "[6/26] build the causally aligned V3 exogenous store"
uv run --frozen python scripts/build_v3_exogenous_store.py >>"$LOG" 2>&1
log "      exogenous store built (capped at 2019-12-31)"

# ---- 7. availability audit ------------------------------------------------
log "[7/26] AVAILABILITY_AUDIT over deterministically sampled origin dates"
set +e
uv run --frozen python scripts/build_v3_exogenous_store.py --audit-only --sample-size 100 \
  >>"$LOG" 2>&1
AUDIT_STATUS=$?
set -e
log "      availability audit exit=$AUDIT_STATUS"

# ---- 8. ABORT on any causality violation ---------------------------------
if [[ "$AUDIT_STATUS" -ne 0 ]]; then
  log "STOP: an exogenous value was available later than the prediction timestamp."
  log "      The V3 programme must not continue."
  exit 3
fi
log "      no causality violation"

# ---- 9-10. the cheap screen ----------------------------------------------
log "[9/26] X0-X4 Logistic screen: 1D/3D/5D/10D x 2014-2018"
uv run --frozen python scripts/run_v3_exogenous_screen.py --model LOGISTIC \
  --config "$CONFIGS/precovid_exogenous_base.yaml" --reset-ledger >>"$LOG" 2>&1
log "      Logistic screen complete"
log "[10/26] X0-X4 HistGradientBoosting screen: 1D/3D/5D/10D x 2014-2018"
uv run --frozen python scripts/run_v3_exogenous_screen.py --model HIST_GRADIENT_BOOSTING \
  --config "$CONFIGS/precovid_exogenous_base.yaml" >>"$LOG" 2>&1
log "      HistGradientBoosting screen complete"

# ---- 11. summarise + common-sample increments ----------------------------
log "[11/26] common-sample incremental metrics"
summarize
log "[12/26] apply the FEATURE-FAMILY GATE"
PASSING="$(summary_value 'summary["horizons_signal_passing"]')"
log "      family/horizon pairs that SIGNAL-PASS: $PASSING"

if [[ "$PASSING" == "[]" ]]; then
  log "[STOP] NO EXOGENOUS INFORMATION FAMILY PASSED THE GATE"
  log "       No neural model is trained. 2019 is NOT opened."
  log "       Recommended next action: BUILD_HISTORICAL_SENTIMENT_EVENT_DATASET"
  summarize
  log "=== PROGRAMME COMPLETE (hard stop, gate failed) ==="
  grep V3_EXOGENOUS_SIGNAL "$RESULTS/V3_EXOGENOUS_REPORT.md" | tail -1
  exit 0
fi

# ---- 13. select at most TOP 2 candidates ---------------------------------
SELECTED="$(summary_value 'json.dumps(summary["selected_neural_candidates"])')"
log "[13/26] selected neural candidate(s): $SELECTED"

run_candidate() {           # run_candidate <family> <horizon> <architecture>
  local family="$1" horizon="$2" architecture="$3"
  local config="$CONFIGS/shared_lstm.yaml"
  [[ "$architecture" == "LSTM_TRANSFORMER_EXOGENOUS" ]] && \
    config="$CONFIGS/lstm_transformer.yaml"
  uv run --frozen python scripts/run_v3_exogenous_neural.py \
    --config "$CONFIGS/precovid_exogenous_base.yaml" --family "$family" \
    --horizon "$horizon" --architecture "$architecture" >>"$LOG" 2>&1
}

# ---- 14. shared LSTM exogenous -------------------------------------------
log "[14/26] SHARED_LSTM_EXOGENOUS on each selected candidate"
while read -r FAMILY HORIZON; do
  [[ -z "$FAMILY" ]] && continue
  log "      $FAMILY @ $HORIZON"
  run_candidate "$FAMILY" "$HORIZON" SHARED_LSTM_EXOGENOUS
done < <(uv run --frozen python -c "
import json, sys
for item in json.loads(sys.argv[1]):
    print(item['family'], item['horizon'])
" "$SELECTED" 2>>"$LOG")

# ---- 15. neural gate ------------------------------------------------------
log "[15/26] apply the NEURAL GATE"
summarize
NEURAL_OK="$(summary_value 'bool(summary["neural"])')"
log "      neural fits recorded: $NEURAL_OK"
if [[ "$NEURAL_OK" != "True" ]]; then
  log "[STOP] no neural exogenous candidate was fitted"
  log "       2019 is NOT opened."
  summarize
  log "=== PROGRAMME COMPLETE (hard stop, no neural candidate) ==="
  grep V3_EXOGENOUS_SIGNAL "$RESULTS/V3_EXOGENOUS_REPORT.md" | tail -1
  exit 0
fi

# ---- 16-17. ablation + seed stability ------------------------------------
log "[16/26] LSTM_TRANSFORMER_EXOGENOUS ablation (must earn its complexity)"
while read -r FAMILY HORIZON; do
  [[ -z "$FAMILY" ]] && continue
  run_candidate "$FAMILY" "$HORIZON" LSTM_TRANSFORMER_EXOGENOUS
done < <(uv run --frozen python -c "
import json, sys
for item in json.loads(sys.argv[1])[:1]:
    print(item['family'], item['horizon'])
" "$SELECTED" 2>>"$LOG")

log "[17/26] seed stability with seeds: $SEEDS"
read -r WIN_FAMILY WIN_HORIZON WIN_ARCH <<<"$(uv run --frozen python - <<'PY' 2>>"$LOG"
import json
from pathlib import Path
summary = json.loads(Path("results/v3/pre_covid_exogenous/v3_exogenous_summary.json").read_text())
candidates = summary["selected_neural_candidates"]
first = candidates[0]
print(first["family"], first["horizon"], "SHARED_LSTM_EXOGENOUS")
PY
)"
log "      winner candidate: $WIN_FAMILY @ $WIN_HORIZON ($WIN_ARCH)"
for SEED in $SEEDS; do
  uv run --frozen python scripts/run_v3_exogenous_neural.py \
    --config "$CONFIGS/precovid_exogenous_base.yaml" --family "$WIN_FAMILY" \
    --horizon "$WIN_HORIZON" --architecture "$WIN_ARCH" --seeds "$SEED" >>"$LOG" 2>&1
done
summarize
STABLE="$(summary_value '(summary.get("seed_stability") or {}).get("stable")')"
log "      seed stability stable=$STABLE"
if [[ "$STABLE" != "True" ]]; then
  log "[STOP] the winner is UNSTABLE across seeds; 2019 is NOT opened."
  summarize
  log "=== PROGRAMME COMPLETE (hard stop, unstable winner) ==="
  grep V3_EXOGENOUS_SIGNAL "$RESULTS/V3_EXOGENOUS_REPORT.md" | tail -1
  exit 0
fi

# ---- 18. freeze -----------------------------------------------------------
log "[18/26] freeze the exogenous model and verify the frozen hashes"
uv run --frozen python scripts/summarize_v3_exogenous.py \
  --config "$CONFIGS/precovid_exogenous_base.yaml" --freeze \
  --family "$WIN_FAMILY" --horizon "$WIN_HORIZON" --architecture "$WIN_ARCH" >>"$LOG" 2>&1
log "      frozen: $RESULTS/frozen_exogenous_model.json"

# ---- 19. ONE 2019 lockbox ------------------------------------------------
log "[19/26] ONE 2019 lockbox run (V3_PRECOVID_LOCKBOX=1, seed 42, exactly once)"
V3_PRECOVID_LOCKBOX=1 uv run --frozen python scripts/run_v3_precovid_lockbox.py \
  --config "$CONFIGS/precovid_exogenous_base.yaml" --family "$WIN_FAMILY" \
  --horizon "$WIN_HORIZON" --architecture "$WIN_ARCH" >>"$LOG" 2>&1
log "      2019 lockbox scored ONCE"

# ---- 20. final report -----------------------------------------------------
log "[20/26] final report including the 2019 lockbox"
summarize --include-lockbox
log "=== PROGRAMME COMPLETE ==="
grep V3_EXOGENOUS_SIGNAL "$RESULTS/V3_EXOGENOUS_REPORT.md" | tail -1