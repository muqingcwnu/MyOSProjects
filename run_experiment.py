"""Run training then full evaluation; writes CSVs to results_fixed/."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

RESULTS_DIR = ROOT / "results_fixed"


def _run(script: str) -> None:
    result = subprocess.run(
        [sys.executable, script],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr)
        raise SystemExit(result.returncode)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Dandelion-Learn experiment")
    print(f"Output: {RESULTS_DIR}")
    print("  1) Training simulation")
    print("  2) Evaluation (performance, ablation, scalability, security)")

    t0 = time.time()
    print("\nStep 1: Training")
    _run("scripts/real_training_simulation.py")
    print(f"[OK] Training done in {(time.time() - t0) / 60:.1f} min")

    t1 = time.time()
    print("\nStep 2: Evaluation")
    _run("run_complete_evaluation.py")
    print(f"[OK] Evaluation done in {(time.time() - t1) / 60:.1f} min")

    csvs = sorted(RESULTS_DIR.glob("*.csv"))
    print(f"\n[OK] {len(csvs)} CSV files in {RESULTS_DIR}")
    for f in csvs:
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
