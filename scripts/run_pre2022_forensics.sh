#!/usr/bin/env bash
# =============================================================================
# PRE-2022 FORENSIC DIAGNOSIS -- unattended master run
#
# The faithful reconstruction (author-confirmed 10 epochs) scores BELOW the
# majority-class baseline on pre-2022 validation, so conventional
# hyperparameter search would only tune noise. This run instead probes the
# STRUCTURAL choices that could explain a high historical result:
#
#   A  adjusted vs unadjusted data          DATA_ADJUSTMENT
#   B  legitimate baseline panel            MODEL_FORM
#   C  pooled (not per-stock) model         PER_STOCK_VS_POOLED
#   D  pre-2022 85/15 chronological analog  SPLIT_PROTOCOL
#   E  target alignment audit (raw OHLCV)   TARGET_ALIGNMENT
#   L  deliberately invalid probes L0-L5    LEAKAGE / OFF_BY_ONE
#   +  confidence subsets, aggregation, label balance
#
# ABSOLUTE CONSTRAINTS (enforced in code, asserted per experiment)
#   * no experiment may score a target date >= 2022-01-01
#   * FINAL_TEST is never set and run_recovered_paper.py is never invoked
#   * Stage B / C / D / E are never run and no config is frozen
#   * PAPER_REFERENCE is never imported; no published metric is a target
#   * the per-stock methodology is never replaced; pooled/85-15 are labelled
#     diagnostics
# =============================================================================

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
export AGENTIC_OUTPUT_ROOT="${AGENTIC_OUTPUT_ROOT:-$RESEARCH_ROOT/output/agentic-forecaster}"
export AGENTIC_DATA_ROOT="${AGENTIC_DATA_ROOT:-$RESEARCH_ROOT/dataset}"
export UV_PROJECT_ENVIRONMENT="$AGENTIC_PROJECT_ROOT/.venv"
export PATH="$RESEARCH_ROOT/tools/bin:$AGENTIC_PROJECT_ROOT/.venv/bin:$PATH"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
# Belt and braces: make a protected year impossible to select even if a future
# edit tried. The firewall also enforces this in code.
export AGENTIC_FIREWALL_START="${AGENTIC_FIREWALL_START:-2022-01-01}"

CONFIG="configs/reproduction_search/paper_snapshot_2025_unadjusted_perf.yaml"
DEVICE="${DEVICE:-auto}"
LOG_DIR="$AGENTIC_OUTPUT_ROOT/reproduction_recovery/logs"
LOCK="$LOG_DIR/pre2022_forensics.lock"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/pre2022_forensics.log"
: >"$LOG"

log() { printf '%s  %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "$LOG"; }

# ---- single-instance lock -------------------------------------------------
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "ERROR: a forensic run is already in progress (lock: $LOCK)" >&2
  exit 75
fi
echo "$$" >&9

trap 'log "FAILED at line $LINENO"; exit 1' ERR

log "=== pre-2022 forensic diagnosis start ==="
log "config: $CONFIG"
log "device: $DEVICE"

# ---- 0. environment + lock-in the toolchain --------------------------------
log "[0/10] environment"
uv sync --all-extras --frozen >>"$LOG" 2>&1
uv run --frozen python scripts/verify_environment.py >>"$LOG" 2>&1
log "      environment verified"

# ---- 1. static analysis + tests -------------------------------------------
log "[1/10] ruff"
uv run --frozen python -m ruff check . >>"$LOG" 2>&1
log "[2/10] pytest"
uv run --frozen python -m pytest -p no:warnings -q >>"$LOG" 2>&1
log "      lint + tests green"

# ---- 2. confirm the author-confirmed Stage-A anchor exists ----------------
log "[3/10] verifying Stage-A T10-F2 artifacts"
uv run --frozen python - <<'PYEOF' >>"$LOG" 2>&1
import json, sys
from pathlib import Path
from agentic_forecaster.config import get_env_roots
root = Path(get_env_roots()["AGENTIC_OUTPUT_ROOT"]) / "reproduction_recovery"
hits = []
for d in sorted(root.glob("EXP-*")):
    f = d / "aggregate_validation_metrics.json"
    if not f.is_file():
        continue
    r = json.loads(f.read_text())["resolved"]
    if r.get("max_epochs") == 10 and r.get("feature_family") == "F2" \
            and r.get("volume_mode") == "raw" and r.get("calibration") == "none":
        hits.append((d.name, r["search_fold"]))
folds = {f for _, f in hits}
# T10-F2 is 2 families -> the F2 half is 3 folds x 8 stocks = 24 fits, i.e.
# 3 experiment runs. F1 is a separate family and is not the anchor here.
if len(hits) < 3 or len(folds) < 3:
    sys.exit(f"FATAL: expected >=3 T10-F2 unadjusted runs across all 3 folds, got {len(hits)} {folds}")
print(f"  Stage-A T10-F2 anchor present: {len(hits)} runs, folds {sorted(folds)}")
PYEOF
log "      Stage-A anchor verified"

# ---- 3. the diagnostics ----------------------------------------------------
log "[4/10] Diagnostics A/B/C/D/E/L"
uv run --frozen python scripts/run_pre2022_forensic_diagnostics.py \
  --config "$CONFIG" --diagnostic all --device "$DEVICE" >>"$LOG" 2>&1
log "      diagnostics complete"

# ---- 4. the offline analyses ---------------------------------------------
log "[5/10] confidence subsets + aggregation + label balance"
uv run --frozen python scripts/summarize_pre2022_forensics.py >>"$LOG" 2>&1
log "      analysis complete"

# ---- 5. firewall re-assertion on every artifact written --------------------
log "[6/10] firewall re-verification"
uv run --frozen python - <<'PYEOF' >>"$LOG" 2>&1
import csv, json, sys
from pathlib import Path
CUT = "2022-01-01"
repo = Path("results/reproduction_recovery/forensics")
run = Path(__import__("os").environ["AGENTIC_OUTPUT_ROOT"]) / "reproduction_recovery" / "forensics"
bad = []
for base in (repo, run):
    for f in base.rglob("*.csv"):
        try:
            rows = list(csv.DictReader(f.open()))
        except Exception:
            continue
        for row in rows[:100000]:
            for k, v in row.items():
                if v and k and k.lower() in ("target_date", "val_last", "train_last",
                                             "date", "sequence_final_date") \
                        and len(v) >= 10 and v[4] == "-" and v[:4].isdigit():
                    if v >= CUT:
                        bad.append((str(f), k, v))
if bad:
    sys.exit(f"FATAL: protected date in forensic artifact: {bad[:5]}")
print("  no protected date found in any forensic artifact")
PYEOF
log "      firewall verified"

# ---- 6. summary ------------------------------------------------------------
log "[7/10] final summary"
uv run --frozen python - <<'PYEOF' >>"$LOG" 2>&1
import json
from pathlib import Path
s = json.loads(Path("results/reproduction_recovery/forensics/pre2022_forensics_summary.json").read_text())
print(f"  demonstrated   : {s['demonstrated_conclusion']}")
print(f"  recommendation : {s['recommendation']}  (NOT executed)")
for c in s["cause_ranking"]:
    print(f"    [{c['status']:<12}] {c['cause']:<24} {c['verdict']:<24} "
          f"delta={c['accuracy_delta_vs_legitimate']}")
print("  " + s["historical_causation"])
PYEOF
log "=== pre-2022 forensic diagnosis complete ==="
log "artifacts : $AGENTIC_OUTPUT_ROOT/reproduction_recovery/forensics/"
log "summary   : results/reproduction_recovery/forensics/pre2022_forensics_summary.json"
