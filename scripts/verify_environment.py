#!/usr/bin/env python3
"""Verify the Agentic Forecaster runtime environment."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    status = "PASS" if condition else "FAIL"
    msg = f"[{status}] {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    if condition:
        PASS += 1
    else:
        FAIL += 1


def check_import(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


def main() -> int:
    print("=== Agentic Forecaster Environment Verification ===\n")

    py_ver = sys.version_info
    check(
        "Python 3.11",
        py_ver.major == 3 and py_ver.minor == 11,
        f"{py_ver.major}.{py_ver.minor}.{py_ver.micro}",
    )

    research_root = os.environ.get("RESEARCH_ROOT", "")
    check("RESEARCH_ROOT set", bool(research_root), research_root)

    project_root = Path(__file__).resolve().parents[1]
    venv_python = project_root / ".venv" / "bin" / "python"
    check(
        "project .venv python",
        venv_python.exists(),
        str(venv_python),
    )

    in_venv = sys.executable.startswith(str(project_root / ".venv"))
    check("running in project venv", in_venv, sys.executable)

    under_research = sys.executable.startswith(research_root) if research_root else False
    check("python under RESEARCH_ROOT", under_research, sys.executable)

    required_imports = [
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "torch",
        "shap",
        "matplotlib",
        "joblib",
        "pydantic",
        "yaml",
        "streamlit",
        "reportlab",
        "openai",
    ]
    for mod in required_imports:
        check(f"import {mod}", check_import(mod))

    try:
        import torch

        check("torch imports", True, torch.__version__)
        cuda_avail = torch.cuda.is_available()
        check("CUDA available (informational)", True, f"available={cuda_avail}")
        if cuda_avail:
            gpu_name = torch.cuda.get_device_name(0)
            check("GPU detected", True, gpu_name)
        else:
            print("[INFO] CUDA not available — CPU fallback mode")
    except ImportError:
        check("torch imports", False)

    path_vars = [
        "AGENTIC_RAW_DATA_ROOT",
        "AGENTIC_PROCESSED_DATA_ROOT",
        "AGENTIC_MODEL_ROOT",
        "AGENTIC_OUTPUT_ROOT",
    ]
    for var in path_vars:
        val = os.environ.get(var, "")
        check(f"{var} set", bool(val), val)
        if val and research_root:
            check(f"{var} under RESEARCH_ROOT", val.startswith(research_root), val)

    cache_vars = [
        "PIP_CACHE_DIR",
        "UV_CACHE_DIR",
        "HF_HOME",
        "HF_HUB_CACHE",
        "TORCH_HOME",
        "TRITON_CACHE_DIR",
        "CUDA_CACHE_PATH",
        "VLLM_CACHE_ROOT",
        "MPLCONFIGDIR",
        "TMPDIR",
    ]
    for var in cache_vars:
        val = os.environ.get(var, "")
        if val and research_root:
            check(f"{var} under RESEARCH_ROOT", val.startswith(research_root), val)
        elif val:
            check(f"{var} set", True, val)

    print(f"\n=== Result: {PASS} passed, {FAIL} failed ===")
    return 1 if FAIL > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
