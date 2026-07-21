"""Run performance, ablation, scalability, and security evaluations."""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from evaluation.experimental_evaluation import (
    ExperimentalSetup,
    run_ablation_studies,
    run_performance_evaluation,
    run_scalability_analysis,
    run_security_analysis,
)

RESULTS_DIR = ROOT / "results_fixed"


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    setup = ExperimentalSetup(
        num_training_episodes=2000,
        jobs_per_episode=100,
        num_runs_per_config=5,
    )

    print("Dandelion-Learn evaluation")
    print(f"Output: {RESULTS_DIR}")
    print(
        f"Episodes={setup.num_training_episodes}, "
        f"jobs/episode={setup.jobs_per_episode}, "
        f"runs={setup.num_runs_per_config}"
    )

    total_start = time.time()
    steps = [
        ("Performance", run_performance_evaluation),
        ("Ablation", run_ablation_studies),
        ("Scalability", run_scalability_analysis),
        ("Security", run_security_analysis),
    ]
    for name, fn in steps:
        print(f"\n=== {name} ===")
        t0 = time.time()
        try:
            fn(setup, RESULTS_DIR)
            print(f"[OK] {name} in {(time.time() - t0) / 60:.1f} min")
        except Exception as exc:
            print(f"[ERROR] {name} failed: {exc}")
            traceback.print_exc()

    print(f"\nTotal time: {(time.time() - total_start) / 60:.1f} min")
    print(f"Results: {RESULTS_DIR}")
    for f in sorted(RESULTS_DIR.glob("*.csv")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
