"""Quick smoke test for GNN + RL + optimizer pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dandelion_learn.baseline_schedulers import FIFOScheduler, ShortestJobFirstScheduler
from dandelion_learn.dandelion_learn_sim import (
    ClusterSimulator,
    InvocationJob,
    MultiObjectiveOptimizer,
    RLScheduler,
    build_cluster_from_config,
    load_azure_invocations,
)
from dandelion_learn.experiment_config import ExperimentConfig
from dandelion_learn.gnn_predictor import GNNPredictor
from dandelion_learn.paths import get_azure_trace_dir
from evaluation.simulation_framework import compute_metrics_from_simulator


def _synthetic_jobs(n: int = 400) -> list:
    jobs = []
    t = 0.0
    for i in range(n):
        jobs.append(
            InvocationJob(
                job_id=i,
                func_id=f"func_{i % 12}",
                arrival_time=t,
                input_size=float(1.0 + np.random.exponential(5.0)),
                base_duration=float(5.0 + np.random.exponential(10.0)),
            )
        )
        t += float(np.random.exponential(0.05))
    return jobs


def main() -> None:
    cfg = ExperimentConfig(gnn_epochs=5, max_jobs_per_day=400, seed=42)
    np.random.seed(cfg.seed)

    try:
        trace_dir = get_azure_trace_dir(ROOT / "azure_trace")
        jobs = load_azure_invocations(trace_dir, day=1, max_jobs=cfg.max_jobs_per_day)
        print(f"Using Azure 2019 trace day 1 from {trace_dir} ({len(jobs)} jobs)")
    except Exception as exc:
        print(f"[WARN] Azure trace unavailable ({exc}); using synthetic jobs")
        jobs = _synthetic_jobs(cfg.max_jobs_per_day)

    train, test = jobs[: len(jobs) // 2], jobs[len(jobs) // 2 :]
    hw = build_cluster_from_config(cfg)

    print(f"Loaded {len(jobs)} jobs; train={len(train)} test={len(test)}; hw={len(hw)}")
    pred = GNNPredictor(d_hidden=cfg.gnn_hidden, L=cfg.gnn_layers, device=cfg.device, seed=cfg.seed)
    hist = pred.fit(
        train,
        epochs=cfg.gnn_epochs,
        lr=cfg.gnn_lr,
        batch_size=cfg.gnn_batch_size,
        weight_decay=cfg.gnn_weight_decay,
        verbose=True,
    )
    print(f"GNN final loss={hist.losses[-1]:.4f}")

    configs = [
        ("FIFO", FIFOScheduler(hw), pred, False),
        ("GNN-Only", ShortestJobFirstScheduler(hw, pred), pred, False),
        ("RL-Only", RLScheduler(hw, epsilon=0.1, use_context=False), pred, False),
        (
            "GNN+RL",
            RLScheduler(hw, epsilon=0.1, use_context=True, context_weight=1.0),
            pred,
            False,
        ),
        (
            "Full",
            RLScheduler(hw, epsilon=0.1, use_context=True, context_weight=1.0),
            pred,
            True,
        ),
    ]

    print("\nAblation smoke (same test set):")
    for name, sched, p, use_opt in configs:
        opt = MultiObjectiveOptimizer(1.0, 0.05, 0.05) if use_opt else None
        sim = ClusterSimulator(
            jobs=list(test),
            hw_units=hw,
            predictor=p,
            scheduler=sched,
            optimizer=opt,
            use_optimizer=use_opt,
            optimizer_rl_bias=cfg.optimizer_rl_bias,
            slo_target_ms=cfg.slo_target_ms,
        )
        sim.run()
        m = compute_metrics_from_simulator(sim, name, "smoke")
        print(
            f"  {name:10s}  p99={m.p99_latency_ms:7.3f} ms  "
            f"thr={m.throughput_jobs_per_sec:8.1f} jobs/s  "
            f"SLO={100*(1-m.slo_violation_rate):5.1f}%  "
            f"E/job={m.energy_per_job:.2f}"
        )

    print("\n[OK] Smoke test completed.")


if __name__ == "__main__":
    main()
