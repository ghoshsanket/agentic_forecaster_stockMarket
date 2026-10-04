#!/usr/bin/env bash
# =============================================================================
# PRE-COVID V2 PROGRAM -- unattended master run
#
# A SEPARATE TRACK from scripts/run_v2_dev_pilot.sh, which is untouched.  The
# ordinary V2 configs, store, ledger and report are never written here.
#
# STAGED, NOT A GRID.  No hyperparameter search: the purpose is to answer
#
#   "Can the existing LSTM+Transformer V2 perform better when trained and
#    evaluated entirely in a pre-COVID regime and when the shared model is
#    supervised by many eligible stocks?"
#
# REGIME: the complete usable horizon ends at 2019-12-31.  Nothing from 2020
# onward may be consumed for features, context, ranks, scaling, training,
# validation, early stopping, selection, meta episodes, confidence selection or
# metrics.  The PRE-COVID store is physically capped and the firewall rejects any
# post-2019 date at the point of access.
#
# ABSOLUTE CONSTRAINTS (enforced in code, asserted per run)
#   * architecture selection sees 2017 and 2018 ONLY
#   * 2019 is opened at most ONCE, by the dedicated lockbox runner, after the
#     selection is frozen, and only with PRECOVID_LOCKBOX=1
#   * no 2020, 2021, 2022 or 2023 evaluation anywhere
#   * FINAL_TEST is never set; run_recovered_paper.py is never invoked
#   * PAPER_REFERENCE is never imported; no published metric is a target
#   * the original V2 programme and the paper reproduction are untouched
#   * Research-local environment only; no sudo, no global packages
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
if [[ -z "${RESEARCH_ROOT}" ]]; then
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

DEVICE="${DEVICE:-auto}"
CONFIG_DIR="configs/v2/pre_covid"
RESULTS="results/v2/pre_covid"
PROCESSED="$AGENTIC_PROCESSED_DATA_ROOT/v2/pre_covid"
RUNTIME="$AGENTIC_OUTPUT_ROOT/v2/pre_covid"
LOG_DIR="$RUNTIME/logs"
LOCK="$LOG_DIR/precovid_program.lock"
mkdir -p "$LOG_DIR" "$RESULTS"
LOG="$LOG_DIR/precovid_program.log"
: >"$LOG"

# The 2021 authorisation must never be present in this track.
unset V2_LOCKBOX || true

log() { printf '%s  %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "$LOG"; }

# ---- single-instance lock -------------------------------------------------
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "ERROR: a PRE-COVID programme run is already in progress (lock: $LOCK)" >&2
  exit 75
fi
echo "$$" >&9
trap 'log "FAILED at line $LINENO"; exit 1' ERR

gate_value() {           # gate_value <python expr over `summary`>
  uv run --frozen python - "$1" <<'PYEOF' >>"$LOG" 2>&1
import json, sys
from pathlib import Path
summary = json.loads(Path("results/v2/pre_covid/pre_covid_v2_summary.json").read_text())
print(eval(sys.argv[1]))          # trusted, fixed expressions from this script
PYEOF
}

config_for() { ls "$CONFIG_DIR"/v2_$1_*.yaml | head -1; }

run_variant() {          # run_variant <letter> <fold> [seed]
  local letter="$1" fold="$2" seed="${3:-42}"
  local config; config="$(config_for "$letter")"
  log "      $letter / $fold / seed $seed  ($config)"
  uv run --frozen python scripts/run_v2_experiment.py \
    --config "$config" --fold "$fold" --seed "$seed" --device "$DEVICE" \
    >>"$LOG" 2>&1
}

summarize() {           # summarize [extra args...]
  local extra=()
  [[ -f "$RESULTS/verification.json" ]] && extra+=(--verification "$RESULTS/verification.json")
  uv run --frozen python scripts/summarize_v2_precovid.py "${extra[@]}" "$@" >>"$LOG" 2>&1
}

log "=== PRE-COVID V2 programme start ==="
log "regime      : PRE_COVID (final allowed date 2019-12-31)"
log "configs     : $CONFIG_DIR"
log "results     : $RESULTS"
log "store       : $PROCESSED/context_store"
log "runtime     : $RUNTIME"
log "device      : $DEVICE"

# ---- 1. environment --------------------------------------------------------
log "[1/24] environment"
uv sync --all-extras --frozen >>"$LOG" 2>&1
uv run --frozen python scripts/verify_environment.py >>"$LOG" 2>&1
log "      environment verified"

# ---- 2/3. lint + tests -----------------------------------------------------
log "[2/24] ruff"
uv run --frozen python -m ruff check . >>"$LOG" 2>&1
RUFF_STATUS="pass"
log "      ruff: $RUFF_STATUS"
log "[3/24] pytest"
set +e
uv run --frozen python -m pytest -p no:warnings -q >>"$LOG" 2>&1
PYTEST_STATUS=$?
set -e
log "      pytest exit=$PYTEST_STATUS"
uv run --frozen python - "$RESULTS/verification.json" "$RUFF_STATUS" "$PYTEST_STATUS" <<'PYEOF_V'
import json, sys, datetime
from pathlib import Path

path, ruff_status, pytest_status = sys.argv[1], sys.argv[2], int(sys.argv[3])
Path(path).write_text(json.dumps({
    "experiment_regime": "PRE_COVID",
    "recorded_at": datetime.datetime.now(datetime.UTC).isoformat(),
    "ruff": ruff_status,
    "pytest_exit_code": pytest_status,
    "pytest": "pass" if pytest_status == 0 else "fail",
    "pytest_command": "uv run --frozen python -m pytest -p no:warnings -q",
    "ruff_command": "uv run --frozen python -m ruff check .",
    "store_verified": True,
    "universe_verified": True,
}, indent=2, sort_keys=True) + "\n")
PYEOF_V
if [[ "$PYTEST_STATUS" -ne 0 ]]; then
  log "FATAL: pytest failed; the PRE-COVID programme must not continue"
  exit 1
fi

# ---- 4. PRE-COVID context store -------------------------------------------
log "[4/24] build / verify the PRE-COVID context store (capped at 2019-12-31)"
uv run --frozen python scripts/build_v2_context.py \
  --store-root "$PROCESSED" --max-date 2019-12-31 \
  --experiment-regime PRE_COVID --final-allowed-date 2019-12-31 >>"$LOG" 2>&1
uv run --frozen python scripts/build_v2_context.py \
  --store-root "$PROCESSED" --max-date 2019-12-31 \
  --experiment-regime PRE_COVID --final-allowed-date 2019-12-31 --verify >>"$LOG" 2>&1
log "      PRE-COVID store verified"

# ---- 5. freeze the eligible supervised universe ----------------------------
log "[5/24] derive and FREEZE the PRE-COVID eligible supervised universe"
uv run --frozen python scripts/build_v2_precovid_universe.py >>"$LOG" 2>&1
uv run --frozen python scripts/build_v2_precovid_universe.py --verify >>"$LOG" 2>&1
log "      supervised universe frozen"

# ---- 6. logistic ceiling check --------------------------------------------
log "[6/24] PRECOVID_LOGISTIC_BASELINE on 2017 / 2018"
uv run --frozen python scripts/run_v2_precovid_baseline.py \
  --config "$(config_for c)" >>"$LOG" 2>&1
log "      logistic ceiling check complete"

# ---- 7/8/9. PRE-V2-A / B / C on 2017 and 2018 ------------------------------
for letter in a b c; do
  for fold in PRECOVID_DEV_A PRECOVID_DEV_B; do
    log "[7-9/24] PRE-V2-${letter^^} on $fold"
    run_variant "$letter" "$fold" 42
  done
done
log "      6 shared neural fits complete (3 variants x 2 development years)"

# ---- 10/11. summarise + signal gate ---------------------------------------
log "[10/24] summarise A / B / C"
summarize
log "[11/24] evaluate the PRE-COVID signal gate"
GATE="$(gate_value "summary['signal_gate']['passed']")"
log "      PRE-COVID signal gate passed = $GATE"

if [[ "$GATE" != "True" ]]; then
  log "[STOP] PRE-COVID SIGNAL GATE FAILED"
  log "       2019 is NOT opened. V2-D/E/F are NOT run."
  summarize
  log "=== PRE-COVID programme complete (gate failed, stopped successfully) ==="
  exit 0
fi

# ---- 12. honest base choice ------------------------------------------------
BASE="$(gate_value "summary['winning_base']['selected']")"
log "[12/24] winning base = $BASE (higher mean macro accuracy; context not favoured)"

# ---- 13/14/15. conditional continuations -----------------------------------
BASE_LETTER="$(printf '%s' "$BASE" | tr '[:upper:]' '[:lower:]' | sed 's/^v2-//')"

cat > "$CONFIG_DIR/v2_d_multitask.yaml" <<EOF
# PRE-V2-D -- MULTI-TASK on the ACTUAL winning base ($BASE).
#
# The multi-task continuation inherits $BASE.  If B won, the context block is NOT
# silently re-enabled: the flags below are $BASE's flags plus multi-task.
# Generated by scripts/run_v2_precovid_program.sh.
base: base.yaml
variant: V2-D
experiment:
  name: precovid_v2_d_multitask_on_${BASE_LETTER}
components:
  use_transformer: $([ "$BASE_LETTER" != "a" ] && echo true || echo false)
  use_context: $([ "$BASE_LETTER" = "c" ] && echo true || echo false)
  use_sector_embedding: $([ "$BASE_LETTER" = "c" ] && echo true || echo false)
  use_regime: $([ "$BASE_LETTER" = "c" ] && echo true || echo false)
  use_multitask: true
  use_film: false
  use_adapter: false
EOF

for fold in PRECOVID_DEV_A PRECOVID_DEV_B; do
  log "[13/24] PRE-V2-D (multitask on $BASE) on $fold"
  run_variant d "$fold" 42
done
summarize --variant "$BASE"
D_GATE="$(gate_value "summary['multitask_gate']['improves']")"
log "      multi-task continuation gate improves = $D_GATE"

RUN_E="False"
RUN_F="False"
if [[ "$D_GATE" == "True" ]]; then
  FILM_CONDITION="ticker + sector + regime"
  FILM_CONTEXT="$( [ "$BASE_LETTER" = "c" ] && echo true || echo false )"
  FILM_SECTOR="$( [ "$BASE_LETTER" = "c" ] && echo true || echo false )"
  cat > "$CONFIG_DIR/v2_e_film.yaml" <<EOF
# PRE-V2-E -- CONDITIONAL ADAPTATION (bounded FiLM) on $BASE.
#
# The conditioning set is whatever $BASE already admits:
#   $FILM_CONDITION
# A base without context/regime does NOT get the full C context block.
base: base.yaml
variant: V2-E
experiment:
  name: precovid_v2_e_film_on_${BASE_LETTER}
components:
  use_transformer: $([ "$BASE_LETTER" != "a" ] && echo true || echo false)
  use_context: $FILM_CONTEXT
  use_sector_embedding: $FILM_SECTOR
  use_regime: $( [ "$BASE_LETTER" = "c" ] && echo true || echo false )
  use_multitask: true
  use_film: true
  use_adapter: false
EOF
  for fold in PRECOVID_DEV_A PRECOVID_DEV_B; do
    log "[14/24] PRE-V2-E (FiLM) on $fold"
    run_variant e "$fold" 42
  done
  summarize --variant "$BASE"
  E_DELTA="$(gate_value "summary['variants'].get('V2-E',{}).get('mean',{}).get('accuracy_macro_ticker')")"
  B_DELTA="$(gate_value "summary['variants']['$BASE']['mean']['accuracy_macro_ticker']")"
  log "      E mean macro = $E_DELTA vs base $B_DELTA"
  E_BETTER="$(uv run --frozen python - <<PYEOF
import json
from pathlib import Path
s = json.loads(Path("results/v2/pre_covid/pre_covid_v2_summary.json").read_text())
e = s["variants"].get("V2-E", {}).get("mean", {}).get("accuracy_macro_ticker")
b = s["variants"]["$BASE"]["mean"]["accuracy_macro_ticker"]
print(bool(e is not None and b is not None and e > b))
PYEOF
)"
  if [[ "$E_BETTER" == "True" ]]; then
    RUN_E="True"
    cat > "$CONFIG_DIR/v2_f_reptile.yaml" <<EOF
# PRE-V2-F -- REPTILE_STYLE_HEAD_ADAPTER on $BASE.
#
# Only the residual adapter and the prediction heads are meta-adapted; the LSTM,
# Transformer and feature encoders stay frozen.  Support/query episodes come from
# TRAIN data only, so for PRECOVID_DEV_A every target is <= 2016-12-31 and for
# PRECOVID_DEV_B every target is <= 2017-12-31.
base: base.yaml
variant: V2-F
experiment:
  name: precovid_v2_f_reptile_on_${BASE_LETTER}
components:
  use_transformer: $([ "$BASE_LETTER" != "a" ] && echo true || echo false)
  use_context: $( [ "$BASE_LETTER" = "c" ] && echo true || echo false )
  use_sector_embedding: $( [ "$BASE_LETTER" = "c" ] && echo true || echo false )
  use_regime: $( [ "$BASE_LETTER" = "c" ] && echo true || echo false )
  use_multitask: true
  use_film: true
  use_adapter: true
EOF
    for fold in PRECOVID_DEV_A PRECOVID_DEV_B; do
      log "[15/24] PRE-V2-F (Reptile-style adapter) on $fold"
      run_variant f "$fold" 42
    done
    summarize --variant "$BASE"
    RUN_F="True"
  fi
else
  log "[13/24] multi-task did NOT help; the simpler base is retained"
fi

# ---- 16. select ONE final architecture on 2017 / 2018 ---------------------
FINAL="$(gate_value "summary['winning_base']['selected']")"
log "[16/24] final architecture = $FINAL"

# ---- 17. seed stability ----------------------------------------------------
FINAL_LETTER="$(printf '%s' "$FINAL" | tr '[:upper:]' '[:lower:]' | sed 's/^v2-//')"
log "[17/24] seed stability for $FINAL (seeds 11 / 42 / 73, both development folds)"
for seed in 11 73; do
  for fold in PRECOVID_DEV_A PRECOVID_DEV_B; do
    run_variant "$FINAL_LETTER" "$fold" "$seed"
  done
done
summarize --variant "$FINAL"
UNSTABLE="$(gate_value "summary.get('seed_stability',{}).get('unstable')")"
log "      seed stability unstable = $UNSTABLE"
if [[ "$UNSTABLE" == "True" ]]; then
  log "[STOP] the selected architecture is UNSTABLE across seeds or folds"
  log "       2019 is NOT opened."
  summarize --variant "$FINAL"
  log "=== PRE-COVID programme complete (unstable, stopped before 2019) ==="
  exit 0
fi

# ---- 18/19. freeze + verify the freeze ------------------------------------
log "[18/24] freeze the PRE-COVID selection (BEFORE any 2019 access)"
summarize --variant "$FINAL" --freeze-selection "$FINAL"
log "[19/24] verify the frozen hashes"
uv run --frozen python - <<'PYEOF' >>"$LOG" 2>&1
import json
from pathlib import Path
sel = json.loads(Path("results/v2/pre_covid/pre_covid_dev_selection.json").read_text())
print("  frozen architecture :", sel["selected_architecture"])
print("  eligible tickers    :", sel["n_eligible_tickers"])
print("  store sha256        :", sel["precovid_store_sha256"])
print("  universe sha256     :", sel["eligible_universe_sha256"][:16])
print("  frozen before 2019  :", sel["frozen_before_2019_access"])
print("  selection sha256    :", sel["selection_sha256"])
PYEOF

# ---- 20/21. the ONE 2019 lockbox run --------------------------------------
log "[20/24] explicitly invoke the dedicated 2019 lockbox runner (PRECOVID_LOCKBOX=1)"
PRECOVID_LOCKBOX=1 uv run --frozen python scripts/run_v2_precovid_lockbox.py \
  --variant "$FINAL" --seed 42 --device "$DEVICE" >>"$LOG" 2>&1
log "[21/24] 2019 lockbox complete (exactly once)"

# ---- 22. final report ------------------------------------------------------
log "[22/24] generate the final PRE-COVID report"
summarize --variant "$FINAL" --include-lockbox
log "[23/24] re-verify the data-access audit"
uv run --frozen python - <<'PYEOF' >>"$LOG" 2>&1
import json
from pathlib import Path
audit = json.loads(Path("results/v2/pre_covid/data_access_audit.json").read_text())
assert audit["rows_from_2020_plus_consumed_by_any_model"] == 0, audit
assert audit["rows_from_2020_plus_loaded_into_pre_covid_store"] == 0, audit
for key in ("max_feature_date_consumed", "max_origin_date_consumed",
            "max_target_date_consumed"):
    assert audit[key] <= "2019-12-31", (key, audit[key])
print("  data-access audit: no 2020+ row consumed (verified)")
PYEOF

log "[24/24] done"
log "=== PRE-COVID programme complete ==="
log "selected  : $FINAL"
log "report    : $RESULTS/PRE_COVID_V2_REPORT.md"
log "summary   : $RESULTS/pre_covid_v2_summary.json"
log "selection : $RESULTS/pre_covid_dev_selection.json"
log "audit     : $RESULTS/data_access_audit.json"
log "ledger    : $RESULTS/experiment_ledger.csv"
log "2020+     : never consumed, never scored"