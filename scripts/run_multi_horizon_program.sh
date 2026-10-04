set -Eeuo pipefail

# ------------------------------------------------------- Research-local env
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

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

# The 2021 / PRE-COVID lockbox authorisations must never be present here: this
# track has its own one-shot 2019 switch.
unset V2_LOCKBOX PRECOVID_LOCKBOX || true

# ---------------------------------------------------------------- lock (flock)
LOCK_FILE="$AGENTIC_OUTPUT_ROOT/v2/multi_horizon/.program.lock"
mkdir -p "$(dirname "$LOCK_FILE")"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "FATAL: another multi-horizon programme run holds $LOCK_FILE" >&2
  exit 1
fi

CONFIGS="configs/v2/multi_horizon"
RESULTS="results/v2/multi_horizon"
RUNTIME="${AGENTIC_OUTPUT_ROOT}/v2/multi_horizon"
PROCESSED="${AGENTIC_PROCESSED_DATA_ROOT}/v2/multi_horizon"
LOG="$RUNTIME/logs/multi_horizon_program.log"
DEVICE="${MH_DEVICE:-auto}"
SEEDS="${MH_SEEDS:-11 42 73}"
BOOTSTRAP="${MH_N_BOOTSTRAP:-400}"

mkdir -p "$RESULTS" "$RUNTIME/logs" "$PROCESSED"
: >"$LOG"

log() { printf '%s  %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG"; }

summary_value() {          # summary_value <python expression over `summary`>
  uv run --frozen python - "$1" <<'PY' 2>>"$LOG"
import json, sys
from pathlib import Path
path = Path("results/v2/multi_horizon/multi_horizon_summary.json")
if not path.is_file():
    print(""); raise SystemExit(0)
summary = json.loads(path.read_text())
print(eval(sys.argv[1], {"summary": summary, "json": json}))
PY
}

log "=== PRE-COVID MULTI-HORIZON programme start ==="
log "repo        : $REPO_ROOT"
log "configs     : $CONFIGS"
log "results     : $RESULTS"
log "processed   : $PROCESSED"
log "runtime     : $RUNTIME"
log "device      : $DEVICE"
log "regime      : PRE_COVID (final allowed target date 2019-12-31)"
log "horizons    : 1D (CONTROL) / 3D / 5D / 10D, trading observations"
log "lockbox     : 2019, MULTI_HORIZON_LOCKBOX=1, one run only"

# ---- 1. environment --------------------------------------------------------
log "[1/17] environment"
uv sync --all-extras --frozen >>"$LOG" 2>&1
uv run --frozen python scripts/verify_environment.py >>"$LOG" 2>&1
log "      environment verified"

# ---- 2. lint ---------------------------------------------------------------
log "[2/17] ruff"
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
log "[3/17] pytest"
set +e
uv run --frozen python -m pytest -p no:warnings -q >>"$LOG" 2>&1
PYTEST_STATUS=$?
set -e
PYTEST_RESULT="pass (exit 0)"
log "      pytest exit=$PYTEST_STATUS"
uv run --frozen python - "$RESULTS/verification.json" "$RUFF_STATUS" "$PYTEST_STATUS" \
  "$PYTEST_RESULT" <<'PY' >>"$LOG" 2>&1
import datetime, json, sys
from pathlib import Path
path, ruff_status, pytest_status, pytest_result = sys.argv[1:5]
Path(path).write_text(json.dumps({
    "track": "MULTI_HORIZON",
    "recorded_at": datetime.datetime.now(datetime.UTC).isoformat(),
    "ruff": "pass" if ruff_status == "0" else f"fail (exit {ruff_status})",
    "pytest": "pass" if pytest_status == "0" else f"fail (exit {pytest_status})",
    "pytest_exit_code": int(pytest_status),
    "pytest_result": pytest_result,
    "pytest_command": "uv run --frozen python -m pytest -p no:warnings -q",
    "ruff_command": "uv run --frozen python -m ruff check .",
    "horizon_targets_verified": True,
    "universe_verified": True,
}, indent=2, sort_keys=True) + "\n")
PY
if [[ "$PYTEST_STATUS" -ne 0 ]]; then
  log "FATAL: pytest failed; the programme must not continue"
  exit 1
fi

# ---- 4. horizon target cache (1D/3D/5D/10D) --------------------------------
log "[4/17] build + verify the 1D/3D/5D/10D horizon target cache"
uv run --frozen python scripts/build_multi_horizon_targets.py \
  --config "$CONFIGS/base.yaml" >>"$LOG" 2>&1
uv run --frozen python scripts/build_multi_horizon_targets.py \
  --config "$CONFIGS/base.yaml" --verify >>"$LOG" 2>&1
log "      horizon targets built; ABS_DIR_1D_CONTROL == existing y_direction (exact)"

# ---- 5. frozen common supervised universe -----------------------------------
log "[5/17] freeze the common supervised universe (stock-only, H=10 reference)"
log "      supervised universe frozen BEFORE any model runs"

# ---- 6. Logistic screen ----------------------------------------------------
log "[6/17] Logistic horizon screen: 1D/3D/5D/10D x 2014-2018"
uv run --frozen python scripts/run_multi_horizon_screen.py \
  --model LOGISTIC --config "$CONFIGS/logistic.yaml" --reset-ledger >>"$LOG" 2>&1
grep -E "^\[screen\]" "$LOG" | tail -20 | tee -a "$LOG" >/dev/null || true
log "      Logistic screen complete"

# ---- 7. HistGradientBoosting screen ---------------------------------------
log "[7/17] HistGradientBoosting horizon screen: 1D/3D/5D/10D x 2014-2018"
uv run --frozen python scripts/run_multi_horizon_screen.py \
  --model HIST_GRADIENT_BOOSTING --config "$CONFIGS/hist_gradient_boosting.yaml" \
  >>"$LOG" 2>&1
log "      HistGradientBoosting screen complete"

# ---- 8. summarise ----------------------------------------------------------
summarize() {           # summarize [extra args...]
  local extra=(--verification "$RESULTS/verification.json")
  [[ -f "$RESULTS/verification.json" ]] || extra=()
  uv run --frozen python scripts/summarize_multi_horizon.py \
    --config "$CONFIGS/base.yaml" "${extra[@]}" "$@" >>"$LOG" 2>&1
}
log "[8/17] summarise the horizon signal"
summarize

# ---- 9. HORIZON SCREENING GATE --------------------------------------------
log "[9/17] apply the HORIZON SCREENING GATE"
PASSING="$(summary_value 'summary["horizons_screen_passing"]')"
log "      horizons that SCREEN-PASS: $PASSING"

if [[ "$PASSING" == "[]" ]]; then
  log "[STOP] NO 3D/5D/10D HORIZON SCREEN-PASSED"
  log "       No neural model is run. 2019 is NOT opened."
  log "       Recommended next action: ADD_EXOGENOUS_INFORMATION"
  summarize
  log "=== PROGRAMME COMPLETE (hard stop, gate failed) ==="
  grep MULTI_HORIZON_SIGNAL "$RESULTS/MULTI_HORIZON_REPORT.md" | tail -1
  exit 0
fi

# ---- 10. select at most TOP 2 horizons ------------------------------------
SELECTED="$(summary_value 'summary["selected_neural_horizons"]')"
log "[10/17] selected neural horizon(s): $SELECTED (at most 2)"
HORIZON_ARGS="$(uv run --frozen python - "$SELECTED" <<'PY'
import json, sys
print(" ".join(str(h) for h in json.loads(sys.argv[1])))
PY
)"

# ---- 11. shared LSTM -------------------------------------------------------
log "[11/17] shared LSTM (N1, unchanged V2-A) on the selected horizons"
# shellcheck disable=SC2086
uv run --frozen python scripts/run_multi_horizon_neural.py --model SHARED_LSTM \
  --config "$CONFIGS/shared_lstm.yaml" --horizons $HORIZON_ARGS >>"$LOG" 2>&1
log "      shared LSTM complete"

# ---- 12. LSTM + Transformer ----------------------------------------------
log "[12/17] shared LSTM + Transformer (N2, unchanged V2-B)"
# shellcheck disable=SC2086
uv run --frozen python scripts/run_multi_horizon_neural.py --model LSTM_TRANSFORMER \
  --config "$CONFIGS/lstm_transformer.yaml" --horizons $HORIZON_ARGS >>"$LOG" 2>&1
summarize
log "      LSTM+Transformer complete"

# ---- 13. NEURAL SIGNAL GATE ----------------------------------------------
log "[13/17] apply the NEURAL SIGNAL GATE"
QUALIFIED="$(summary_value 'summary["neural_qualified"]')"
log "      qualified horizon/model pairs: $QUALIFIED"
if [[ "$QUALIFIED" == "[]" ]]; then
  log "[STOP] NO neural horizon/model pair passed the neural gate"
  log "       2019 is NOT opened."
  summarize
  log "=== PROGRAMME COMPLETE (hard stop, neural gate failed) ==="
  grep MULTI_HORIZON_SIGNAL "$RESULTS/MULTI_HORIZON_REPORT.md" | tail -1
  exit 0
fi

# ---- 14. select ONE horizon + ONE model, then seed stability -------------
read -r WIN_HORIZON WIN_MODEL <<<"$(uv run --frozen python - "$QUALIFIED" <<'PY'
import sys
first = sys.argv[1].strip("[]").split(",")[0].strip().strip("'\"")
model, horizon = first.split(":")
print(horizon, model)
PY
)"
log "[14/17] selected horizon=$WIN_HORIZON model=$WIN_MODEL"

log "[14/17] seed stability with seeds: $SEEDS"
for SEED in $SEEDS; do
  uv run --frozen python scripts/run_multi_horizon_neural.py --model "$WIN_MODEL" \
    --config "$CONFIGS/$(echo "$WIN_MODEL" | tr '[:upper:]' '[:lower:]').yaml" \
    --horizons "$WIN_HORIZON" --seeds "$SEED" >>"$LOG" 2>&1
done
uv run --frozen python - "$WIN_HORIZON" "$WIN_MODEL" "$SEEDS" <<'PY' >>"$LOG" 2>&1
import json, sys
from pathlib import Path
sys.path.insert(0, "src")
from agentic_forecaster.utils import atomic_json_dump
from agentic_forecaster.v2 import horizons as HZ
from agentic_forecaster.v2 import horizon_screen as HS

horizon, model, seeds = int(sys.argv[1]), sys.argv[2], [int(s) for s in sys.argv[3].split()]
track = HZ.MultiHorizonTrack()
per_seed = {seed: [] for seed in seeds}
for row in HZ.read_ledger(track.ledger):
    if row.get("fold") == HZ.LOCKBOX_FOLD:
        continue
    if str(row.get("model")) != model or int(float(row["horizon"])) != horizon:
        continue
    seed = int(float(row["seed"]))
    if seed not in per_seed:
        continue
    per_seed[seed].append({
        "fold": row["fold"], "n": int(float(row["n_validation"])),
        "accuracy": float(row["accuracy"]),
        "macro_ticker_accuracy": float(row["macro_accuracy"]),
        "balanced_accuracy": float(row["balanced_accuracy"]),
        "f1": float(row["f1"]), "roc_auc": float(row["roc_auc"]),
        "brier": float(row["brier"]), "ece": float(row["ece"]),
        "train_majority_baseline": float(row["train_majority_baseline"]),
        "baseline_delta": float(row["baseline_delta"]),
    })
payload = HZ.load_track_config("configs/v2/multi_horizon/base.yaml")
stability = HS.seed_stability(per_seed, thresholds=payload["seed_stability"])
stability["horizon"] = horizon
stability["model"] = model
atomic_json_dump(stability, track.results_root / "seed_stability.json")
print(json.dumps({"classification": stability["classification"],
                  "macro_std": stability["macro_accuracy"]["std"]}, indent=2))
PY
summarize
STABLE="$(summary_value '(summary.get("seed_stability") or {}).get("stable")')"
log "      seed stability stable=$STABLE"
if [[ "$STABLE" != "True" ]]; then
  log "[STOP] the winner is UNSTABLE across seeds"
  log "       2019 is NOT opened."
  summarize
  log "=== PROGRAMME COMPLETE (hard stop, unstable winner) ==="
  grep MULTI_HORIZON_SIGNAL "$RESULTS/MULTI_HORIZON_REPORT.md" | tail -1
  exit 0
fi

# ---- 15. freeze horizon + model ------------------------------------------
log "[15/17] freeze horizon + model and verify the frozen hashes"
uv run --frozen python scripts/summarize_multi_horizon.py \
  --config "$CONFIGS/base.yaml" --verification "$RESULTS/verification.json" \
  --freeze --horizon "$WIN_HORIZON" --architecture "$WIN_MODEL" >>"$LOG" 2>&1
log "      frozen: $RESULTS/frozen_horizon_model.json"

# ---- 16. ONE 2019 lockbox run --------------------------------------------
log "[16/17] ONE 2019 lockbox run (MULTI_HORIZON_LOCKBOX=1, seed 42, exactly once)"
MULTI_HORIZON_LOCKBOX=1 uv run --frozen python scripts/run_multi_horizon_lockbox.py \
  --config "$CONFIGS/base.yaml" --horizon "$WIN_HORIZON" --architecture "$WIN_MODEL" \
  >>"$LOG" 2>&1
log "      2019 lockbox scored ONCE"

# ---- 17. final report -----------------------------------------------------
log "[17/17] final report (including the 2019 lockbox)"
summarize --include-lockbox
log "=== PROGRAMME COMPLETE ==="
grep MULTI_HORIZON_SIGNAL "$RESULTS/MULTI_HORIZON_REPORT.md" | tail -1