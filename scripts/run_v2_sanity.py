#!/usr/bin/env python3
"""V2 sanity gate: synthetic learnability and the shuffled-label control.

Both checks run BEFORE any real V2 training:

``--mode learnable``
    A deterministic, learnable relationship between the last input row and the
    next-day direction. The V2 model must learn it far above chance.

``--mode shuffled``
    The same task with the TRAIN labels shuffled. Validation accuracy must fall
    back to approximately chance, which proves the model is using its inputs.

``--mode both`` (default)
    Both, written to ``results/v2/sanity/synthetic_learnable.json`` and
    ``results/v2/sanity/shuffled_labels.json``.

If the learnable check FAILS the real V2 programme must stop before spending
compute on real data: every later number would be uninterpretable.

No real market data is used, and neither check can touch the paper-test firewall
because the synthetic dates are arbitrary and no V2 store is read.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentic_forecaster.utils import atomic_json_dump, ensure_dir, setup_logging
from agentic_forecaster.v2.sanity import run_synthetic_check, sanity_payload

RESULTS_RELATIVE = Path("results/v2/sanity")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="both",
                        choices=["learnable", "shuffled", "both"])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    setup_logging()
    out_dir = args.out_dir or (REPO_ROOT / RESULTS_RELATIVE)
    ensure_dir(out_dir)

    results = []
    if args.mode in ("learnable", "both"):
        result = run_synthetic_check("synthetic_learnable", seed=args.seed,
                                     epochs=args.epochs, device=args.device)
        atomic_json_dump(result.to_dict(), out_dir / "synthetic_learnable.json")
        results.append(result)
        print(f"synthetic_learnable: validation_accuracy="
              f"{result.validation_accuracy_micro:.4f} passed={result.passed}")

    if args.mode in ("shuffled", "both"):
        result = run_synthetic_check("shuffled_labels", shuffle_labels=True,
                                     seed=args.seed, epochs=args.epochs,
                                     device=args.device)
        atomic_json_dump(result.to_dict(), out_dir / "shuffled_labels.json")
        results.append(result)
        print(f"shuffled_labels:     validation_accuracy="
              f"{result.validation_accuracy_micro:.4f} passed={result.passed}")

    payload = sanity_payload(results)
    atomic_json_dump(payload, out_dir / "sanity_summary.json")
    print(json.dumps({"all_passed": payload["all_passed"],
                      "gate": payload["gate"]}, indent=2))
    if not payload["all_passed"]:
        print("FATAL: a V2 sanity check failed; STOP before real V2 training",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())