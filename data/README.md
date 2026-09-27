# Data

## Raw Data Location

```
$RESEARCH_ROOT/dataset/agentic-forecaster/raw
```

## Processed Data Location

```
$RESEARCH_ROOT/dataset/agentic-forecaster/processed
```

The Git repository intentionally does NOT contain the full dataset.

## Source

- **Kaggle URL:** https://www.kaggle.com/datasets/debashis74017/algo-trading-data-nifty-100-data-with-indicators
- **Slug:** `debashis74017/algo-trading-data-nifty-100-data-with-indicators`

## Download

```bash
source $RESEARCH_ROOT/scripts/research-env.sh
python scripts/download_dataset.py
```

## Environment Variables

| Variable | Purpose |
|---|---|
| `AGENTIC_RAW_DATA_ROOT` | Raw downloaded data |
| `AGENTIC_PROCESSED_DATA_ROOT` | Processed sequences |
| `AGENTIC_DATA_ROOT` | Parent data directory |

## Inspection

```bash
python scripts/inspect_raw_dataset.py
```

Writes metadata to `$AGENTIC_DATA_ROOT/metadata/`.
