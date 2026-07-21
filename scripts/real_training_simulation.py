"""Train GNN + RL on Azure traces and write metrics CSVs to results_fixed/."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dandelion_learn.dandelion_learn_sim import (
    ClusterSimulator,
    InvocationJob,
    MultiObjectiveOptimizer,
    RLScheduler,
    build_cluster_from_config,
    load_azure_invocations,
)
from dandelion_learn.experiment_config import DEFAULT_CONFIG
from dandelion_learn.gnn_predictor import GNNPredictor
from dandelion_learn.paths import get_azure_trace_dir


class RealTrainingLogger:
    """Collects GNN and RL training metrics."""

    def __init__(self) -> None:
        self.gnn_epochs: List[int] = []
        self.gnn_losses: List[float] = []
        self.gnn_runtime_errors: List[float] = []
        self.gnn_memory_errors: List[float] = []
        self.rl_episodes: List[int] = []
        self.rl_rewards: List[float] = []
        self.rl_q_values: Dict[int, List[float]] = {}
        self.rl_latencies: List[float] = []
        self.episode_latencies: List[float] = []
        self.episode_throughputs: List[float] = []
        self.episode_gpu_utils: List[float] = []

    def log_gnn_training(
        self, epoch: int, loss: float, runtime_error: float, memory_error: float
    ) -> None:
        self.gnn_epochs.append(epoch)
        self.gnn_losses.append(loss)
        self.gnn_runtime_errors.append(runtime_error)
        self.gnn_memory_errors.append(memory_error)

    def log_rl_update(
        self, episode: int, hw_id: int, q_value: float, reward: float, latency: float
    ) -> None:
        self.rl_episodes.append(episode)
        self.rl_rewards.append(reward)
        self.rl_q_values.setdefault(hw_id, []).append(q_value)
        self.rl_latencies.append(latency)

    def log_episode_performance(
        self, latency: float, throughput: float, gpu_util: float
    ) -> None:
        self.episode_latencies.append(latency)
        self.episode_throughputs.append(throughput)
        self.episode_gpu_utils.append(gpu_util)


def compute_prediction_error(
    predictor: GNNPredictor, jobs: List[InvocationJob], hw_type: str = "cpu"
) -> Tuple[float, float]:
    runtime_errors: List[float] = []
    memory_errors: List[float] = []
    for job in jobs[:100]:
        pred = predictor.predict(job, hw_type=hw_type)
        actual_runtime = job.base_duration
        runtime_errors.append(
            abs(actual_runtime - pred.runtime_ms) / (actual_runtime + 1e-6)
        )
        actual_memory = job.input_size * 1.5
        memory_errors.append(
            abs(actual_memory - pred.mem_mb) / (actual_memory + 1e-6)
        )
    return float(np.mean(runtime_errors)), float(np.mean(memory_errors))


def real_training_simulation(
    root: Path,
    num_training_episodes: int = 2000,
    jobs_per_episode: int = 100,
) -> RealTrainingLogger:
    logger = RealTrainingLogger()
    cfg = DEFAULT_CONFIG

    trace_dir = get_azure_trace_dir(root / "azure_trace")
    print(f"Azure trace dir: {trace_dir}")
    all_jobs: List[InvocationJob] = []
    for day in [1, 2, 3]:
        all_jobs.extend(load_azure_invocations(trace_dir, day=day, max_jobs=2000))

    np.random.seed(cfg.seed)
    np.random.shuffle(all_jobs)
    train_size = int(len(all_jobs) * 0.7)
    train_jobs = all_jobs[:train_size]
    eval_jobs = all_jobs[train_size:]

    hw_units = build_cluster_from_config(cfg)
    predictor = GNNPredictor(
        d_hidden=cfg.gnn_hidden, L=cfg.gnn_layers, device=cfg.device, seed=cfg.seed
    )
    scheduler = RLScheduler(
        hw_units,
        epsilon=cfg.rl_epsilon,
        use_context=True,
        context_weight=cfg.rl_context_weight,
        queue_weight=cfg.rl_queue_weight,
        cold_weight=cfg.rl_cold_weight,
        cold_start_ms=cfg.cold_start_ms,
    )
    optimizer = MultiObjectiveOptimizer(
        alpha=cfg.optimizer_alpha,
        beta=cfg.optimizer_beta,
        gamma=cfg.optimizer_gamma,
        queue_weight=cfg.optimizer_queue_weight,
        cold_weight=cfg.optimizer_cold_weight,
        cold_start_ms=cfg.cold_start_ms,
    )

    print("Initial GNN training on historical data...")
    hist = predictor.fit(
        train_jobs,
        epochs=cfg.gnn_epochs,
        lr=cfg.gnn_lr,
        batch_size=cfg.gnn_batch_size,
        weight_decay=cfg.gnn_weight_decay,
        verbose=True,
    )
    for ep, loss, rt_mae, mem_mae in zip(
        hist.epochs, hist.losses, hist.runtime_mae, hist.memory_mae
    ):
        logger.log_gnn_training(ep, loss, rt_mae, mem_mae)

    runtime_err, memory_err = compute_prediction_error(predictor, train_jobs)
    print(f"  Final pred error: runtime={runtime_err:.4f}, memory={memory_err:.4f}")

    print(f"Running {num_training_episodes} training episodes...")
    for episode in range(num_training_episodes):
        episode_jobs = np.random.choice(
            eval_jobs,
            size=min(jobs_per_episode, len(eval_jobs)),
            replace=False,
        ).tolist()

        sim = ClusterSimulator(
            jobs=list(episode_jobs),
            hw_units=hw_units,
            predictor=predictor,
            optimizer=optimizer,
            scheduler=scheduler,
            use_optimizer=True,
            optimizer_rl_bias=cfg.optimizer_rl_bias,
            slo_target_ms=cfg.slo_target_ms,
            enable_cold_start=cfg.enable_cold_start,
            cold_start_ms=cfg.cold_start_ms,
        )
        start = time.time()
        sim.run()
        wall_time = time.time() - start
        sim_time = sim.makespan if sim.makespan > 0 else wall_time

        p99_latency = 0.0
        throughput = 0.0
        gpu_util = 0.0

        if sim.job_latencies:
            latencies_ms = np.array(sim.job_latencies) * 1000.0
            avg_latency = float(np.mean(latencies_ms))
            p99_latency = float(np.percentile(latencies_ms, 99))
            throughput = len(episode_jobs) / sim_time if sim_time > 0 else 0.0
            gpu_units = [hw for hw in hw_units if hw.hw_type == "gpu"]
            total_gpu_time = sum(sim.hw_utilization.get(hw.hw_id, 0) for hw in gpu_units)
            gpu_util = total_gpu_time / sim_time if sim_time > 0 else 0.0
            logger.log_episode_performance(p99_latency, throughput, gpu_util)
            for hw_id in range(len(hw_units)):
                logger.log_rl_update(
                    episode,
                    hw_id,
                    float(scheduler.values[hw_id]),
                    -avg_latency,
                    p99_latency,
                )

        if episode > 0 and episode % 20 == 0:
            recent_jobs = episode_jobs + train_jobs[:200]
            ft = predictor.fit(
                recent_jobs,
                epochs=3,
                lr=cfg.gnn_lr,
                batch_size=cfg.gnn_batch_size,
                weight_decay=cfg.gnn_weight_decay,
                verbose=False,
            )
            if ft.losses:
                logger.log_gnn_training(
                    episode // 20,
                    ft.losses[-1],
                    ft.runtime_mae[-1],
                    ft.memory_mae[-1],
                )

        if episode > 0 and episode % 100 == 0:
            print(
                f"  Episode {episode}/{num_training_episodes}: "
                f"p99={p99_latency:.2f}ms, throughput={throughput:.2f} jobs/s, "
                f"GPU util={gpu_util:.2%}"
            )

    print("[OK] Training simulation completed")
    return logger


def save_training_csvs(logger: RealTrainingLogger, results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "epoch": logger.gnn_epochs,
            "loss": logger.gnn_losses,
            "runtime_error": logger.gnn_runtime_errors,
            "memory_error": logger.gnn_memory_errors,
        }
    ).to_csv(results_dir / "gnn_training.csv", index=False)

    pd.DataFrame(
        {
            "episode": logger.rl_episodes,
            "reward": logger.rl_rewards,
            "latency": logger.rl_latencies,
        }
    ).to_csv(results_dir / "rl_training.csv", index=False)

    pd.DataFrame(logger.rl_q_values).to_csv(results_dir / "rl_q_values.csv", index=False)

    pd.DataFrame(
        {
            "episode": range(len(logger.episode_latencies)),
            "p99_latency": logger.episode_latencies,
            "throughput": logger.episode_throughputs,
            "gpu_utilization": logger.episode_gpu_utils,
        }
    ).to_csv(results_dir / "performance_metrics.csv", index=False)


def main() -> None:
    root = ROOT
    results_dir = root / "results_fixed"
    print("=" * 70)
    print("TRAINING SIMULATION")
    print("=" * 70)
    logger = real_training_simulation(
        root,
        num_training_episodes=DEFAULT_CONFIG.num_training_episodes,
        jobs_per_episode=DEFAULT_CONFIG.jobs_per_episode,
    )
    save_training_csvs(logger, results_dir)
    print(f"[OK] Training CSVs saved to {results_dir}")


if __name__ == "__main__":
    main()
