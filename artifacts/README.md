# Artifacts

## What is included

| Path | Description |
|---|---|
| `manifests/models/runtime_checkpoint_index.json` | SHA-256 hashes and byte sizes of all trained checkpoints from the author's run |
| `models/` | Canonical exported checkpoints (Git LFS) — see policy below |
| `SUBMISSION_MANIFEST.md` | Full submission audit manifest |

## Checkpoint policy

The full canonical 50-model checkpoint set is **not** committed to normal
Git.  Instead:

1. **Code** defining all models is committed under `src/agentic_forecaster/models/`.
2. **Manifests** with checkpoint hashes are committed under
   `artifacts/manifests/models/`.
3. **Reproduction commands** regenerate the checkpoints exactly:
   ```bash
   python scripts/train_all.py --config configs/paper.yaml
   ```

## Git LFS

`.gitattributes` configures Git LFS tracking for `*.pt`, `*.pth`, `*.ckpt`,
`*.joblib`, `*.safetensors` and `*.onnx`.  If you choose to commit canonical
checkpoints under `artifacts/models/`, they will be stored via LFS.

**Check whether Git LFS is installed:**

```bash
git lfs version
```

**Expected LFS size:** to be determined after the first full training run.
The per-checkpoint size for the Attention-LSTM is approximately
`(input_size * hidden_size + hidden_size * hidden_size * num_layers +
hidden_size * num_classes) * 4 bytes` — on the order of tens of KB per model,
so the full 50-model set should be a few MB and suitable for LFS.

## How to verify checkpoint hashes

```bash
# Verify a checkpoint against the manifest
sha256sum <checkpoint>.pt
# Compare against artifacts/manifests/models/runtime_checkpoint_index.json
```

## How omitted checkpoints can be recreated

```bash
# Train all 50 models from scratch
python scripts/train_all.py --config configs/paper.yaml

# Or run the full reproduction pipeline
python scripts/reproduce_paper.py --config configs/paper.yaml --export-final-results
```

Checkpoints are written under `$AGENTIC_MODEL_ROOT` (Category B, outside Git).

## Runtime directory

The original full artifacts live under::

    $AGENTIC_MODEL_ROOT
    (default: $RESEARCH_ROOT/models/agentic-forecaster/)

This path is environment-specific; the manifests and reproduction commands
in this repository are the portable, authoritative record.
