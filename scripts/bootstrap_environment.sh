#!/usr/bin/env bash
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_PROJECT_ROOT="$(cd "$_SCRIPT_DIR/.." && pwd)"

if [ -f "$_PROJECT_ROOT/../../scripts/research-env.sh" ]; then
    # shellcheck source=../../../scripts/research-env.sh
    source "$_PROJECT_ROOT/../../scripts/research-env.sh"
elif [ -f "$RESEARCH_ROOT/scripts/research-env.sh" ]; then
    # shellcheck source=/dev/null
    source "$RESEARCH_ROOT/scripts/research-env.sh"
fi

echo "=== Agentic Forecaster Environment Bootstrap ==="
echo "RESEARCH_ROOT: $RESEARCH_ROOT"
echo "PROJECT_ROOT:  $_PROJECT_ROOT"
echo

if [ -z "${RESEARCH_ROOT:-}" ]; then
    echo "ERROR: RESEARCH_ROOT is not set."
    exit 1
fi

if [ ! -d "$RESEARCH_ROOT" ]; then
    echo "ERROR: RESEARCH_ROOT does not exist: $RESEARCH_ROOT"
    exit 1
fi

PYTHON_BIN="$RESEARCH_ROOT/runtimes/python-bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
    echo "ERROR: Research-local Python not found: $PYTHON_BIN"
    exit 1
fi

PY_VER="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [ "$PY_VER" != "3.11" ]; then
    echo "ERROR: Python 3.11 required, found $PY_VER"
    exit 1
fi
echo "[PASS] Python 3.11: $PY_VER"

if ! command -v uv &>/dev/null; then
    echo "ERROR: uv not found in PATH"
    exit 1
fi
UV_VER="$(uv --version)"
echo "[PASS] uv: $UV_VER"

VENV_DIR="$_PROJECT_ROOT/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "[INFO] Creating .venv..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

VENV_PY="$VENV_DIR/bin/python"
if [ ! -x "$VENV_PY" ]; then
    echo "ERROR: .venv python not found: $VENV_PY"
    exit 1
fi
echo "[PASS] .venv python: $VENV_PY"

echo "[INFO] Syncing dependencies from uv.lock..."
cd "$_PROJECT_ROOT"
uv sync --all-extras --frozen

echo "[INFO] Verifying imports..."
"$VENV_PY" -c "
import numpy, pandas, scipy, sklearn, torch, shap, matplotlib, joblib, pydantic, yaml, streamlit, reportlab, openai
print('[PASS] All imports OK')
"

echo "[INFO] Checking PyTorch/CUDA..."
"$VENV_PY" -c "
import torch
print(f'[INFO] torch: {torch.__version__}')
print(f'[INFO] CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'[INFO] GPU: {torch.cuda.get_device_name(0)}')
else:
    print('[INFO] Running in CPU mode')
"

echo "[INFO] Running verify_environment.py..."
"$VENV_PY" "$_PROJECT_ROOT/scripts/verify_environment.py"

echo
echo "=== Bootstrap complete ==="
