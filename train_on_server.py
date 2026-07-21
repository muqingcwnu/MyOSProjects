#!/usr/bin/env python3
"""Server entry: smoke test, evaluation, or full train+eval pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _print_hw() -> None:
    print("Dandelion-Learn server train")
    try:
        import torch

        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("PyTorch not installed. Run: pip install -r requirements.txt")
        sys.exit(1)

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))
    from dandelion_learn.experiment_config import DEFAULT_CONFIG, resolve_device

    print(f"Device: {DEFAULT_CONFIG.device} -> {resolve_device(DEFAULT_CONFIG.device)}")
    print(
        f"GNN epochs={DEFAULT_CONFIG.gnn_epochs}, "
        f"batch={DEFAULT_CONFIG.gnn_batch_size}, "
        f"RL episodes={DEFAULT_CONFIG.num_training_episodes}"
    )


def _run(script: str) -> int:
    print(f"\n>>> python {script}")
    return subprocess.call([sys.executable, str(ROOT / script)], cwd=str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train / evaluate Dandelion-Learn")
    parser.add_argument("--smoke", action="store_true", help="Quick smoke test only")
    parser.add_argument("--full", action="store_true", help="Train + evaluate")
    parser.add_argument("--eval-only", action="store_true", help="Evaluation only")
    args = parser.parse_args()

    _print_hw()
    (ROOT / "results_fixed").mkdir(exist_ok=True)

    if args.smoke or (not args.full and not args.eval_only):
        rc = _run("scripts/smoke_test_pipeline.py")
        if rc != 0:
            sys.exit(rc)
        if not args.full and not args.eval_only:
            print("\n[OK] Smoke done.")
            print("Next: python train_on_server.py --eval-only")
            print("   or: python train_on_server.py --full")
            return

    if args.eval_only:
        t0 = time.time()
        rc = _run("run_complete_evaluation.py")
        print(f"\nEvaluation wall time: {time.time() - t0:.1f}s")
        sys.exit(rc)

    if args.full:
        t0 = time.time()
        rc = _run("run_experiment.py")
        print(f"\nFull pipeline wall time: {time.time() - t0:.1f}s")
        sys.exit(rc)


if __name__ == "__main__":
    main()
