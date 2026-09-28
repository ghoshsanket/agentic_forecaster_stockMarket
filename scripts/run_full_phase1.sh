#!/usr/bin/env bash
#
# run_full_phase1.sh — ONE unattended master run of the Phase-1 pipeline.
#
#   validate -> lint -> test -> (optional real-data test) -> prepare/cache
#   daily data -> RELIANCE real-data smoke test -> FULL 49x2 reproduction
#   -> strict submission validation -> inventory -> summary
#
# Every stage is a hard gate: the script aborts on the first failure and never
# marks the run complete after a failure.
#
# Usage:
#   ./scripts/run_full_phase1.sh
#
# Environment overrides (all optional):
#   RUN_ID               run identifier (default full_phase1_YYYYMMDD_HHMMSS)
#   DEVICE               auto | cpu | cuda | cuda:N   (default auto)
#   RUN_REAL_DATA_TEST   1 to also run the opt-in real-data pytest (default 0)
#   MIN_FREE_GB          abort if free space is below this (default 10)
#
# Deliberately NOT done: git add/commit/push, dataset download, global installs,
# sudo, CUDA reconfiguration, ad-hoc parallel training loops.

set -Eeuo pipefail

# --------------------------------------------------------------------------
# Locate the workspace. Prefer derivation from this script's own location so
# no other user's path is hard-coded.
# --------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESEARCH_ROOT="${RESEARCH_ROOT:-$(cd "$PROJECT_ROOT/../.." && pwd)}"

# Allow AGENTIC_PROJECT_ROOT to be overridden, but verify it points at a repo.
AGENTIC_PROJECT_ROOT="${AGENTIC_PROJECT_ROOT:-$PROJECT_ROOT}"
export RESEARCH_ROOT AGENTIC_PROJECT_ROOT

if [[ ! -f "$RESEARCH_ROOT/scripts/research-env.sh" ]]; then
    echo "FATAL: research-env.sh not found under $RESEARCH_ROOT" >&2
    echo "       Set RESEARCH_ROOT explicitly and retry." >&2
    exit 78
fi

# shellcheck source=/dev/null
source "$RESEARCH_ROOT/scripts/research-env.sh"

# research-env.sh may set AGENTIC_PROJECT_ROOT itself; keep ours if it is valid.
if [[ ! -d "$AGENTIC_PROJECT_ROOT" ]]; then
    AGENTIC_PROJECT_ROOT="$PROJECT_ROOT"
fi
export AGENTIC_PROJECT_ROOT

# Pin the project virtualenv ahead of anything else on PATH.
#
# research-env.sh puts $RESEARCH_ROOT/runtimes/python-bin on PATH, but that
# directory may expose only `python` and not `python3`; in that case `python3`
# can still resolve to a global interpreter (observed: miniforge Python 3.13).
# Prepending the project venv makes `python` AND `python3` deterministic no
# matter how this script was launched.  All project work additionally goes
# through `uv run`, which uses this same locked environment.
if [[ -d "$AGENTIC_PROJECT_ROOT/.venv/bin" ]]; then
    export PATH="$AGENTIC_PROJECT_ROOT/.venv/bin:$PATH"
fi

# --------------------------------------------------------------------------
# Options
# --------------------------------------------------------------------------
RUN_ID="${RUN_ID:-full_phase1_$(date +%Y%m%d_%H%M%S)}"
DEVICE="${DEVICE:-auto}"
RUN_REAL_DATA_TEST="${RUN_REAL_DATA_TEST:-0}"
MIN_FREE_GB="${MIN_FREE_GB:-10}"
SMOKE_TICKER="${SMOKE_TICKER:-RELIANCE}"
export RUN_ID DEVICE

# --------------------------------------------------------------------------
# Logging (tee to terminal + file, line-buffered)
# --------------------------------------------------------------------------
LOG_DIR="$AGENTIC_OUTPUT_ROOT/logs"
LOCK_DIR="$AGENTIC_OUTPUT_ROOT/locks"
mkdir -p "$LOG_DIR" "$LOCK_DIR"
LOG_FILE="$LOG_DIR/${RUN_ID}.log"
FAILED_MARKER="$LOG_DIR/${RUN_ID}.failed"
COMPLETED_MARKER="$LOG_DIR/${RUN_ID}.completed"
SUMMARY_FILE="$LOG_DIR/${RUN_ID}_summary.txt"
export LOG_FILE FAILED_MARKER COMPLETED_MARKER SUMMARY_FILE

START_EPOCH="$(date +%s)"
START_TS="$(date -Is)"

# Redact anything that looks like a secret before it reaches the log or stdout.
# Every log line passes through this, so no stage can leak credentials.
redact() {
    sed -E \
        -e 's/(KAGGLE_API_TOKEN|KAGGLE_KEY|KAGGLE_USERNAME|LLM_API_KEY|LLM_BASE_URL|OPENAI_API_KEY|WANDB_API_KEY)=.*/\1=<redacted>/g' \
        -e 's/(sk-[A-Za-z0-9]{16,})/<redacted>/g'
}

log()  { printf '%s  %s\n' "$(date -Is)" "$*" | redact | tee -a "$LOG_FILE"; }
sec()  { printf '\n%s\n== %s\n%s\n' "============================================================" "$*" "============================================================" | tee -a "$LOG_FILE"; }
die()  { log "FATAL: $*"; exit 1; }

# SIGPIPE-safe "first N lines".
#
# `| head -N |` in the middle of a pipeline under `set -o pipefail` makes head
# close the pipe early, which sends SIGPIPE (141) to the downstream stage and
# aborts the whole run. awk consumes its entire input, so it never triggers
# SIGPIPE while still printing only the first N lines.
first_n() { awk -v n="${1:-5}" 'NR<=n'; }


on_error() {
    local exit_code=$?
    local line_no="${1:-unknown}"
    local cmd="${2:-unknown}"
    {
        printf '\n'
        printf '============================================================\n'
        printf 'FULL PHASE-1 RUN FAILED\n'
        printf '============================================================\n'
        printf 'run id            : %s\n' "$RUN_ID"
        printf 'failed at line    : %s\n' "$line_no"
        printf 'failing command   : %s\n' "$cmd"
        printf 'exit code         : %s\n' "$exit_code"
        printf 'timestamp         : %s\n' "$(date -Is)"
        printf 'log file          : %s\n' "$LOG_FILE"
        printf '============================================================\n'
    } | tee -a "$LOG_FILE"
    {
        printf 'run_id=%s\n' "$RUN_ID"
        printf 'status=failed\n' 
        printf 'exit_code=%s\n' "$exit_code"
        printf 'failed_at_line=%s\n' "$line_no"
        printf 'failed_command=%s\n' "$cmd"
        printf 'timestamp=%s\n' "$(date -Is)"
        printf 'log_file=%s\n' "$LOG_FILE"
    } | redact > "$FAILED_MARKER"
    rm -f "$COMPLETED_MARKER"
    log "Failure marker written: $FAILED_MARKER"
    log "No completion marker was created. The repository has NOT been declared complete."
    exit "$exit_code"
}
trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR

# --------------------------------------------------------------------------
# Single-instance lock (flock preferred, PID-file fallback)
# --------------------------------------------------------------------------
RUN_LOCK="$LOCK_DIR/full_phase1.lock"
LOCK_MODE=""
if command -v flock >/dev/null 2>&1; then
    exec 9>"$RUN_LOCK"
    if ! flock -n 9; then
        echo "ERROR: another full Phase-1 run appears to be active" >&2
        echo "       lock: $RUN_LOCK" >&2
        echo "       inspect with: ./scripts/check_full_phase1_status.sh" >&2
        exit 75
    fi
    LOCK_MODE="flock"
else
    PIDFILE="$LOCK_DIR/full_phase1.pid"
    if [[ -f "$PIDFILE" ]]; then
        existing="$(cat "$PIDFILE" 2>/dev/null || true)"
        if [[ -n "$existing" ]] && kill -0 "$existing" 2>/dev/null; then
            echo "ERROR: another full Phase-1 run appears to be active (pid $existing)" >&2
            exit 75
        fi
        log "Removing stale lock from pid $existing"
        rm -f "$PIDFILE"
    fi
    echo "$$" > "$PIDFILE"
    trap 'rm -f "$PIDFILE"' EXIT
    LOCK_MODE="pidfile"
fi

sec "PHASE-1 MASTER RUN — $RUN_ID"

# --------------------------------------------------------------------------
# 0. System information (non-secret)
# --------------------------------------------------------------------------
sec "STEP 0 — SYSTEM INFORMATION"
log "run id            : $RUN_ID"
log "start timestamp   : $START_TS"
log "date              : $(date)"
log "hostname          : $(hostname)"
log "whoami            : $(whoami)"
log "pwd               : $(pwd)"
log "uname             : $(uname -a)"
log "device policy     : $DEVICE"
log "real-data pytest  : $RUN_REAL_DATA_TEST"
log "lock mode         : $LOCK_MODE"
log "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset, respected as-is>}"
if command -v nvidia-smi >/dev/null 2>&1; then
    log "nvidia-smi:"
    nvidia-smi 2>&1 | first_n 20 | redact | tee -a "$LOG_FILE" || log "  (nvidia-smi returned nonzero)"
else
    log "WARNING: nvidia-smi not found; continuing (CPU fallback may apply)"
fi

# --------------------------------------------------------------------------
# 1. Workspace verification
# --------------------------------------------------------------------------
sec "STEP 1 — WORKSPACE VERIFICATION"
log "RESEARCH_ROOT           : $RESEARCH_ROOT"
log "AGENTIC_PROJECT_ROOT    : $AGENTIC_PROJECT_ROOT"
log "AGENTIC_RAW_DATA_ROOT   : $AGENTIC_RAW_DATA_ROOT"
log "AGENTIC_PROCESSED_DATA_ROOT: $AGENTIC_PROCESSED_DATA_ROOT"
log "AGENTIC_MODEL_ROOT      : $AGENTIC_MODEL_ROOT"
log "AGENTIC_OUTPUT_ROOT     : $AGENTIC_OUTPUT_ROOT"

[[ -d "$RESEARCH_ROOT" ]]          || die "RESEARCH_ROOT missing: $RESEARCH_ROOT"
[[ -d "$AGENTIC_PROJECT_ROOT" ]]   || die "AGENTIC_PROJECT_ROOT missing"
[[ -d "$AGENTIC_RAW_DATA_ROOT" ]]  || die "AGENTIC_RAW_DATA_ROOT missing (dataset NOT downloaded — not auto-downloading)"
[[ -f "$AGENTIC_PROJECT_ROOT/pyproject.toml" ]] || die "not a project root: no pyproject.toml"
[[ -f "$AGENTIC_PROJECT_ROOT/uv.lock" ]]        || die "missing committed uv.lock"

for p in "$AGENTIC_RAW_DATA_ROOT" "$AGENTIC_PROCESSED_DATA_ROOT" \
         "$AGENTIC_MODEL_ROOT" "$AGENTIC_OUTPUT_ROOT" "$AGENTIC_PROJECT_ROOT"; do
    case "$(cd "$(dirname "$p")" 2>/dev/null && pwd)/$(basename "$p")" in
        "$RESEARCH_ROOT"/*|"$RESEARCH_ROOT") : ;;
        *) die "path escapes RESEARCH_ROOT: $p" ;;
    esac
done
log "all AGENTIC_* paths verified beneath RESEARCH_ROOT"

mkdir -p "$AGENTIC_PROCESSED_DATA_ROOT" "$AGENTIC_MODEL_ROOT" "$AGENTIC_OUTPUT_ROOT"
cd "$AGENTIC_PROJECT_ROOT"
log "working directory       : $(pwd)"

# --------------------------------------------------------------------------
# 2. Tooling + Research-local interpreter
# --------------------------------------------------------------------------
sec "STEP 2 — TOOLING AND PYTHON ISOLATION"
log "git version : $(git --version 2>&1 || echo unavailable)"
log "uv version  : $(uv --version 2>&1 || echo unavailable)"

# Report every python entry point, because a bare `python3` on PATH can belong
# to a global installation even when `python` is Research-local.
for exe_name in python python3; do
    resolved_cmd="$(command -v "$exe_name" 2>/dev/null || true)"
    if [[ -n "$resolved_cmd" ]]; then
        resolved_real="$(readlink -f "$resolved_cmd" 2>/dev/null || echo "$resolved_cmd")"
        if [[ "$resolved_real" == "$RESEARCH_ROOT"* ]]; then
            where="RESEARCH-LOCAL"
        else
            where="GLOBAL (not used for project work)"
        fi
        log "$exe_name on PATH      : $resolved_cmd  [$where]"
    fi
done

if [[ -x "$AGENTIC_PROJECT_ROOT/.venv/bin/python" ]]; then
    PY="$AGENTIC_PROJECT_ROOT/.venv/bin/python"
    log "project venv python   : $PY"
else
    log "WARNING: project .venv not found; relying on uv run to provision it"
    PY=""
fi
log "python --version      : $(python --version 2>&1 || echo unavailable)"

# The interpreter used for the isolation check must be the project venv, and
# must live under RESEARCH_ROOT.  Never install anything otherwise.
if [[ -n "$PY" ]]; then
    resolved="$("$PY" -c 'import sys; print(sys.executable)' 2>/dev/null || true)"
else
    resolved="$(python -c 'import sys; print(sys.executable)' 2>/dev/null || true)"
fi
log "sys.executable        : ${resolved:-unknown}"
if [[ -z "$resolved" ]]; then
    die "could not resolve sys.executable for the isolation check"
fi
if [[ "$resolved" != "$RESEARCH_ROOT"* ]]; then
    die "project interpreter is OUTSIDE RESEARCH_ROOT: $resolved (refusing to install anything)"
fi
log "interpreter is Research-local: OK"

# --------------------------------------------------------------------------
# 3. Disk space
# --------------------------------------------------------------------------
sec "STEP 3 — DISK SPACE"
log "df for $RESEARCH_ROOT:"
df -h "$RESEARCH_ROOT" | redact | tee -a "$LOG_FILE"
avail_kb="$(df -Pk "$RESEARCH_ROOT" | awk 'NR==2 {print $4}')"
avail_gb=$(( avail_kb / 1024 / 1024 ))
log "available free space: ${avail_gb} GB (minimum ${MIN_FREE_GB} GB)"
if (( avail_gb < MIN_FREE_GB )); then
    die "critically low free space (${avail_gb} GB < ${MIN_FREE_GB} GB)"
fi

# --------------------------------------------------------------------------
# 4. Raw dataset present (never download)
# --------------------------------------------------------------------------
sec "STEP 4 — RAW DATASET CHECK"
if [[ ! -d "$AGENTIC_RAW_DATA_ROOT" ]]; then
    die "raw data root missing: $AGENTIC_RAW_DATA_ROOT — this script never downloads the dataset"
fi
raw_count="$(find "$AGENTIC_RAW_DATA_ROOT" -maxdepth 1 -name '*.csv' -type f | wc -l | tr -d ' ')"
raw_size="$(du -sh "$AGENTIC_RAW_DATA_ROOT" 2>/dev/null | cut -f1)"
log "raw CSV files : $raw_count"
log "raw disk usage: $raw_size"
log "sample files  :"
find "$AGENTIC_RAW_DATA_ROOT" -maxdepth 1 -name '*.csv' -type f | sort | first_n 5 | redact | tee -a "$LOG_FILE"
if (( raw_count < 50 )); then
    die "raw dataset looks empty (only $raw_count CSV files) — NOT downloading automatically"
fi
log "raw dataset present: OK"

# --------------------------------------------------------------------------
# 5. Locked dependency sync
# --------------------------------------------------------------------------
sec "STEP 5 — DEPENDENCY SYNC (uv, frozen, Research-local)"
log "running: uv sync --all-extras --frozen"
uv sync --all-extras --frozen 2>&1 | redact | tee -a "$LOG_FILE" | tail -5
log "dependency sync complete"

# --------------------------------------------------------------------------
# 6. Environment verification + torch/CUDA info
# --------------------------------------------------------------------------
sec "STEP 6 — ENVIRONMENT VERIFICATION"
log "running: uv run python scripts/verify_environment.py"
uv run python scripts/verify_environment.py 2>&1 | redact | tee -a "$LOG_FILE" | tail -8
log "running: torch / CUDA information"
uv run python -c "
import torch
print('torch_version=', torch.__version__)
print('torch_cuda_runtime=', torch.version.cuda)
print('cuda_available=', torch.cuda.is_available())
print('cuda_device_count=', torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'gpu_{i}=', torch.cuda.get_device_name(i))
" 2>&1 | redact | tee -a "$LOG_FILE"

# --------------------------------------------------------------------------
# 7. Lint
# --------------------------------------------------------------------------
sec "STEP 7 — LINT"
log "running: uv run ruff check ."
uv run ruff check . 2>&1 | redact | tee -a "$LOG_FILE" | tail -5
log "lint passed"

# --------------------------------------------------------------------------
# 8. Ordinary test suite (real-data tests must skip)
# --------------------------------------------------------------------------
sec "STEP 8 — ORDINARY TEST SUITE"
log "running: uv run pytest  (AGENTIC_RUN_REAL_DATA_TESTS deliberately unset)"
if [[ -n "${AGENTIC_RUN_REAL_DATA_TESTS:-}" ]]; then
    log "NOTE: AGENTIC_RUN_REAL_DATA_TESTS is set in this shell; unsetting for the ordinary run"
fi
PYTEST_RESULT="$(env -u AGENTIC_RUN_REAL_DATA_TESTS uv run pytest 2>&1 | redact | tee -a "$LOG_FILE" | tail -3)"
printf '%s\n' "$PYTEST_RESULT" | redact | tee -a "$LOG_FILE"
log "ordinary pytest finished"

# --------------------------------------------------------------------------
# 9. Optional real-data integration test
# --------------------------------------------------------------------------
sec "STEP 9 — OPTIONAL REAL-DATA INTEGRATION TEST"
if [[ "$RUN_REAL_DATA_TEST" == "1" ]]; then
    log "RUN_REAL_DATA_TEST=1 -> running the opt-in real-data test"
    AGENTIC_RUN_REAL_DATA_TESTS=1 uv run pytest \
        tests/integration/test_reproduction_outputs.py -v 2>&1 \
        | redact | tee -a "$LOG_FILE" | tail -8
    log "opt-in real-data test passed"
else
    log "RUN_REAL_DATA_TEST=0 (default) — skipping (the RELIANCE smoke test below covers real data)"
fi

# --------------------------------------------------------------------------
# 10. Prepare / cache daily data
# --------------------------------------------------------------------------
sec "STEP 10 — PREPARE AND CACHE DAILY DATA"
log "running: uv run python -m agentic_forecaster prepare-data --config configs/paper.yaml"
uv run python -m agentic_forecaster prepare-data \
    --config configs/paper.yaml 2>&1 | redact | tee -a "$LOG_FILE" | tail -20
log "prepare-data completed"

# --------------------------------------------------------------------------
# 11. Verify processed data
# --------------------------------------------------------------------------
sec "STEP 11 — VERIFY PROCESSED DATA"
proc_daily="$AGENTIC_PROCESSED_DATA_ROOT/daily"
if [[ ! -d "$proc_daily" ]]; then
    die "no processed daily directory at $proc_daily"
fi
proc_count="$(find "$proc_daily" -maxdepth 1 -name '*.parquet' -type f | wc -l | tr -d ' ')"
proc_size="$(du -sh "$AGENTIC_PROCESSED_DATA_ROOT" 2>/dev/null | cut -f1)"
log "processed parquet files : $proc_count"
log "processed disk usage    : $proc_size"
log "sample files:"
find "$proc_daily" -maxdepth 1 -name '*.parquet' -type f | sort | first_n 5 | redact | tee -a "$LOG_FILE"
if (( proc_count < 1 )); then
    die "prepare-data produced no processed files"
fi
log "processed data verified: OK"

# --------------------------------------------------------------------------
# 12. Real RELIANCE smoke test (mandatory gate)
# --------------------------------------------------------------------------
sec "STEP 12 — REAL-DATA SMOKE TEST ($SMOKE_TICKER)"
log "running: uv run python scripts/smoke_test_real.py --ticker $SMOKE_TICKER --config configs/paper.yaml --device $DEVICE"
uv run python scripts/smoke_test_real.py \
    --ticker "$SMOKE_TICKER" \
    --config configs/paper.yaml \
    --device "$DEVICE" 2>&1 | redact | tee -a "$LOG_FILE" | tail -30
log "real smoke test passed — safe to proceed to the full reproduction"

# --------------------------------------------------------------------------
# 13. Full Phase-1 reproduction
# --------------------------------------------------------------------------
sec "STEP 13 — FULL PHASE-1 REPRODUCTION"
REPRO_DIR="$AGENTIC_OUTPUT_ROOT/reproduction/$RUN_ID"
if [[ -d "$REPRO_DIR" ]]; then
    die "run directory already exists, refusing to overwrite: $REPRO_DIR (choose a different RUN_ID)"
fi
log "expected coverage is derived by the application from"
log "  results/ticker_availability.csv (available tickers)"
log "  x the paper walk-forward folds — not hard-coded here."
log "running: uv run python -m agentic_forecaster reproduce-paper --config configs/paper.yaml --device $DEVICE --run-id $RUN_ID --export-final-results"
uv run python -m agentic_forecaster reproduce-paper \
    --config configs/paper.yaml \
    --device "$DEVICE" \
    --run-id "$RUN_ID" \
    --export-final-results 2>&1 | redact | tee -a "$LOG_FILE" | tail -40
log "full reproduction finished"

# --------------------------------------------------------------------------
# 14. Strict submission validation
# --------------------------------------------------------------------------
sec "STEP 14 — STRICT SUBMISSION VALIDATION"
log "running: uv run python scripts/validate_submission.py   (strict; no permissive flags)"
uv run python scripts/validate_submission.py 2>&1 | redact | tee -a "$LOG_FILE" | tail -12
log "strict validation passed"

# --------------------------------------------------------------------------
# 15. Result inventory
# --------------------------------------------------------------------------
sec "STEP 15 — RESULT INVENTORY"
if [[ ! -d "$REPRO_DIR" ]]; then
    die "expected reproduction directory missing: $REPRO_DIR"
fi
log "reproduction directory: $REPRO_DIR"
log "  files: $(find "$REPRO_DIR" -maxdepth 1 -type f | wc -l | tr -d ' '), size: $(du -sh "$REPRO_DIR" | cut -f1)"
find "$REPRO_DIR" -maxdepth 1 -type f -printf '    %f  (%s bytes)\n' | sort | redact | tee -a "$LOG_FILE"
for d in figures reports; do
    if [[ -d "$REPRO_DIR/$d" ]]; then
        log "  $d/: $(find "$REPRO_DIR/$d" -type f | wc -l | tr -d ' ') files"
    fi
done
log "repository export:"
for d in results figures reports artifacts/manifests; do
    if [[ -d "$AGENTIC_PROJECT_ROOT/$d" ]]; then
        log "  $d/: $(find "$AGENTIC_PROJECT_ROOT/$d" -type f | wc -l | tr -d ' ') files, $(du -sh "$AGENTIC_PROJECT_ROOT/$d" | cut -f1)"
    fi
done

# --------------------------------------------------------------------------
# 16. Model inventory
# --------------------------------------------------------------------------
sec "STEP 16 — MODEL INVENTORY"
pt_count="$(find "$AGENTIC_MODEL_ROOT" -name '*.pt' -type f 2>/dev/null | wc -l | tr -d ' ')"
pt_size="$(find "$AGENTIC_MODEL_ROOT" -name '*.pt' -type f -printf '%s\n' 2>/dev/null | awk '{s+=$1} END {printf "%.1f MB", (s==""?0:s)/1048576}')"
dir_count="$(find "$AGENTIC_MODEL_ROOT" -mindepth 2 -maxdepth 2 -type d 2>/dev/null | wc -l | tr -d ' ')"
log "trained .pt files        : $pt_count"
log "total .pt size           : $pt_size"
log "ticker/fold directories  : $dir_count"

# --------------------------------------------------------------------------
# 17. Git status (read-only — never add/commit/push)
# --------------------------------------------------------------------------
sec "STEP 17 — GIT STATUS (read-only)"
GIT_SHORT="$AGENTIC_PROJECT_ROOT/.git"
if [[ -d "$GIT_SHORT" ]]; then
    log "git status --short:"
    git -C "$AGENTIC_PROJECT_ROOT" status --short 2>&1 | redact | tee -a "$LOG_FILE" | first_n 60
    log "git diff --stat:"
    git -C "$AGENTIC_PROJECT_ROOT" diff --stat 2>&1 | redact | tee -a "$LOG_FILE" | tail -20
    log "NOTE: no git add / commit / push was performed. Inspect the repository yourself."
else
    log "WARNING: no .git directory found; skipping git status"
fi

# --------------------------------------------------------------------------
# 18. Summary + completion marker
# --------------------------------------------------------------------------
sec "STEP 18 — RUN SUMMARY"
FINISH_TS="$(date -Is)"
FINISH_EPOCH="$(date +%s)"
DURATION=$(( FINISH_EPOCH - START_EPOCH ))
AVAIL_GB_USED="$(df -Pk "$RESEARCH_ROOT" | awk 'NR==2 {print $4}')"

{
    echo "RUN_ID=$RUN_ID"
    echo "START_TIME=$START_TS"
    echo "FINISH_TIME=$FINISH_TS"
    echo "DURATION_SECONDS=$DURATION"
    echo "HOST=$(hostname)"
    echo "USER=$(whoami)"
    echo "PYTHON_VERSION=$(python3 --version 2>&1 || python --version 2>&1)"
    echo "TORCH_VERSION=$(uv run python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo unknown)"
    echo "CUDA_RUNTIME=$(uv run python -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo unknown)"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
    echo "GPU=$(uv run python -c 'import torch;print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu-only")' 2>/dev/null || echo unknown)"
    echo "DEVICE_POLICY=$DEVICE"
    echo "RAW_DATA_ROOT=$AGENTIC_RAW_DATA_ROOT"
    echo "PROCESSED_DATA_ROOT=$AGENTIC_PROCESSED_DATA_ROOT"
    echo "MODEL_ROOT=$AGENTIC_MODEL_ROOT"
    echo "REPRODUCTION_OUTPUT=$REPRO_DIR"
    echo "REPOSITORY=$AGENTIC_PROJECT_ROOT"
    echo "LOG_FILE=$LOG_FILE"
    echo "LOCK_MODE=$LOCK_MODE"
    echo "PYTEST_RESULT=$(printf '%s' "$PYTEST_RESULT" | tail -1)"
    echo "RUFF_RESULT=passed"
    echo "SMOKE_TEST_RESULT=passed ($SMOKE_TICKER)"
    echo "FULL_REPRODUCTION_RESULT=completed"
    echo "STRICT_VALIDATION_RESULT=passed"
    echo "MODEL_FILE_COUNT=$pt_count"
    echo "MODEL_TOTAL_SIZE=$pt_size"
    echo "RAW_CSV_FILE_COUNT=$raw_count"
    echo "PROCESSED_PARQUET_COUNT=$proc_count"
    echo "FREE_SPACE_GB_AT_END=$(( AVAIL_GB_USED / 1024 / 1024 ))"
    echo "GIT_ACTION=none (no add/commit/push)"
} | redact > "$SUMMARY_FILE"
log "summary written: $SUMMARY_FILE"

{
    echo "run_id=$RUN_ID"
    echo "status=completed"
    echo "completed_at=$FINISH_TS"
    echo "duration_seconds=$DURATION"
    echo "reproduction_dir=$REPRO_DIR"
    echo "repository=$AGENTIC_PROJECT_ROOT"
    echo "validation=strict_passed"
    echo "summary=$SUMMARY_FILE"
    echo "log_file=$LOG_FILE"
} | redact > "$COMPLETED_MARKER"
log "completion marker written: $COMPLETED_MARKER"

trap - ERR
sec "FULL PHASE-1 REPRODUCTION COMPLETED SUCCESSFULLY"
cat <<EOF | tee -a "$LOG_FILE"
============================================================
FULL PHASE-1 REPRODUCTION COMPLETED SUCCESSFULLY
============================================================
RUN_ID                : $RUN_ID
LOG FILE              : $LOG_FILE
RUN SUMMARY           : $SUMMARY_FILE
REPRODUCTION DIRECTORY: $REPRO_DIR
REPOSITORY DIRECTORY  : $AGENTIC_PROJECT_ROOT
DURATION              : $DURATION s
============================================================
Inspect ./scripts/check_full_phase1_status.sh for status.

Do not rerun the full experiment unless intentionally starting a new
reproduction run.
============================================================
EOF
