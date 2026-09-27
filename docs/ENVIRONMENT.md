# Environment

## System

- **OS:** Linux (aarch64)
- **Python:** 3.11.16
- **Python executable:** `$AGENTIC_PROJECT_ROOT/.venv/bin/python`
- **uv:** 0.12.19

## NVIDIA / GPU

- **Driver:** 580.142 (system/shared — do not modify)
- **GPU:** NVIDIA GB10
- **CUDA (nvidia-smi):** 13.0
- **PyTorch CUDA runtime:** 13.0
- **PyTorch version:** 2.14.0

The NVIDIA driver and system CUDA are shared system resources. PyTorch and all
Python packages are project-local dependencies installed inside `.venv`.

## Package Versions

| Package | Version |
|---|---|
| numpy | 2.4.6 |
| pandas | 3.0.6 |
| scipy | 1.17.1 |
| scikit-learn | 1.9.1 |
| torch | 2.14.0 |
| shap | 0.51.0 |
| matplotlib | 3.11.2 |
| joblib | 1.6.0 |
| pydantic | 2.13.5 |
| PyYAML | 6.0.3 |
| streamlit | 1.64.0 |
| reportlab | 5.0.1 |
| openai | 3.19.2 |
| pytest | 9.1.1 |
| ruff | 0.16.9 |

## Isolation

All Python packages, caches, and runtime artifacts resolve under
`$RESEARCH_ROOT`. No global or system Python packages are used.

Key environment variables (set by `research-env.sh`):

- `PIP_CACHE_DIR` → `$RESEARCH_ROOT/cache/pip`
- `UV_CACHE_DIR` → `$RESEARCH_ROOT/cache/uv`
- `HF_HOME` → `$RESEARCH_ROOT/models/huggingface`
- `TORCH_HOME` → `$RESEARCH_ROOT/cache/torch`
- `TRITON_CACHE_DIR` → `$RESEARCH_ROOT/cache/triton`
- `MPLCONFIGDIR` → `$RESEARCH_ROOT/cache/matplotlib`
- `TMPDIR` → `$RESEARCH_ROOT/tmp`

## Setup

```bash
source $RESEARCH_ROOT/scripts/research-env.sh
cd $AGENTIC_PROJECT_ROOT
uv sync --all-extras
python scripts/verify_environment.py
```

## Bootstrap

```bash
scripts/bootstrap_environment.sh
```

## CPU Fallback

The project runs on CPU when CUDA is not available. Use `--device auto` to
automatically select CUDA when available, otherwise CPU.
