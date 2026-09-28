#!/usr/bin/env bash
#
# check_full_phase1_status.sh — non-destructive status inspection for the
# Phase-1 master run. This script only READS; it never modifies any file,
# never signals a process, and never starts a run.
#
# Usage:
#   ./scripts/check_full_phase1_status.sh              # discover latest run
#   ./scripts/check_full_phase1_status.sh <RUN_ID>     # inspect a specific run
#
# Exit codes:
#   0  completed (or inspected successfully)
#   1  usage / environment error
#   3  run appears active
#   4  run failed
#   5  run not found

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESEARCH_ROOT="${RESEARCH_ROOT:-$(cd "$PROJECT_ROOT/../.." && pwd)}"
AGENTIC_PROJECT_ROOT="${AGENTIC_PROJECT_ROOT:-$PROJECT_ROOT}"

if [[ -f "$RESEARCH_ROOT/scripts/research-env.sh" ]]; then
    # shellcheck source=/dev/null
    source "$RESEARCH_ROOT/scripts/research-env.sh"
fi
if [[ ! -d "${AGENTIC_OUTPUT_ROOT:-}" ]]; then
    echo "ERROR: could not determine AGENTIC_OUTPUT_ROOT (source research-env.sh first)" >&2
    exit 1
fi

LOG_DIR="$AGENTIC_OUTPUT_ROOT/logs"
LOCK_DIR="$AGENTIC_OUTPUT_ROOT/locks"
MODEL_ROOT="${AGENTIC_MODEL_ROOT:-$RESEARCH_ROOT/models/agentic-forecaster}"

RUN_ID_ARG="${1:-}"

# ---- locate a run id ------------------------------------------------------
RUN_ID="$RUN_ID_ARG"
if [[ -z "$RUN_ID" ]]; then
    if [[ -d "$LOG_DIR" ]]; then
        RUN_ID="$(find "$LOG_DIR" -maxdepth 1 -name 'full_phase1_*.log' -type f -printf '%f\n' 2>/dev/null \
                  | sed 's/\.log$//' | sort | tail -1)"
    fi
fi
if [[ -z "$RUN_ID" ]]; then
    echo "No full_phase1 run found under $LOG_DIR"
    exit 5
fi

LOG_FILE="$LOG_DIR/${RUN_ID}.log"
COMPLETED="$LOG_DIR/${RUN_ID}.completed"
FAILED="$LOG_DIR/${RUN_ID}.failed"
SUMMARY="$LOG_DIR/${RUN_ID}_summary.txt"
LOCK_FILE="$LOCK_DIR/full_phase1.lock"

echo "============================================================"
echo " Phase-1 master run status"
echo "============================================================"
echo "RUN_ID        : $RUN_ID"
echo "log file      : $LOG_FILE"
echo "summary       : $SUMMARY"
echo "inspected at  : $(date -Is)"
echo "host          : $(hostname)"

# ---- active process -------------------------------------------------------
ACTIVE="no"
if [[ -f "$LOCK_FILE" ]]; then
    if command -v flock >/dev/null 2>&1 && flock -n "$LOCK_FILE" true 2>/dev/null; then
        : # lock free -> not active
    else
        ACTIVE="yes"
    fi
fi
if [[ -f "$LOCK_DIR/full_phase1.pid" ]]; then
    pid="$(cat "$LOCK_DIR/full_phase1.pid" 2>/dev/null || true)"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        ACTIVE="yes (pid $pid)"
    fi
fi
echo "active        : $ACTIVE"

# ---- state ----------------------------------------------------------------
STATE="unknown"
if [[ -f "$COMPLETED" ]]; then
    STATE="completed"
elif [[ -f "$FAILED" ]]; then
    STATE="failed"
elif [[ "$ACTIVE" != "no" ]]; then
    STATE="running"
elif [[ -f "$LOG_FILE" ]]; then
    STATE="incomplete (no completed/failed marker; run likely interrupted)"
fi
echo "state         : $STATE"

# ---- latest log lines -----------------------------------------------------
if [[ -f "$LOG_FILE" ]]; then
    echo
    echo "---------------- last 30 log lines ----------------"
    tail -n 30 "$LOG_FILE"
    echo "-------------------------------------------------"
else
    echo
    echo "(no log file yet at $LOG_FILE)"
fi

# ---- current model count --------------------------------------------------
if [[ -d "$MODEL_ROOT" ]]; then
    pt_count="$(find "$MODEL_ROOT" -name '*.pt' -type f 2>/dev/null | wc -l | tr -d ' ')"
    pt_size="$(find "$MODEL_ROOT" -name '*.pt' -type f -printf '%s\n' 2>/dev/null \
               | awk '{s+=$1} END {printf "%.1f MB", (s==""?0:s)/1048576}')"
else
    pt_count=0
    pt_size="0 MB"
fi
echo
echo "models        : $pt_count .pt files, $pt_size total, under $MODEL_ROOT"

# ---- current reproduction output count ------------------------------------
if [[ -d "$AGENTIC_OUTPUT_ROOT/reproduction" ]]; then
    repro_runs="$(find "$AGENTIC_OUTPUT_ROOT/reproduction" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')"
else
    repro_runs=0
fi
specific="$AGENTIC_OUTPUT_ROOT/reproduction/$RUN_ID"
if [[ -d "$specific" ]]; then
    spec_files="$(find "$specific" -maxdepth 1 -type f 2>/dev/null | wc -l | tr -d ' ')"
    spec_size="$(du -sh "$specific" 2>/dev/null | cut -f1)"
else
    spec_files=0
    spec_size="n/a"
fi
echo "reproduction  : $repro_runs run dir(s) under $AGENTIC_OUTPUT_ROOT/reproduction"
echo "this run      : $spec_files file(s), $spec_size, at $specific"

# ---- exit code ------------------------------------------------------------
case "$STATE" in
    completed) exit 0 ;;
    running)   exit 3 ;;
    failed)    exit 4 ;;
    *)         exit 0 ;;
esac
