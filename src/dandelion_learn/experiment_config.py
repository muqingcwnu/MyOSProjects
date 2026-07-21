"""Unified experiment configuration for training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


def resolve_device(requested: str = "auto") -> str:
    """
    Return a usable device string.

    RTX 5090 (sm_120 / Blackwell) needs a very new PyTorch build. Older wheels
    report CUDA as available but fail at runtime — we probe and fall back to CPU.
    """
    req = (requested or "auto").lower()
    if req == "cpu":
        return "cpu"
    try:
        import torch

        if not torch.cuda.is_available():
            return "cpu"

        try:
            x = torch.zeros(1, device="cuda")
            _ = (x + 1).item()
            del x
            torch.cuda.empty_cache()
            return "cuda"
        except Exception as exc:
            print(
                f"[WARN] CUDA not usable with this PyTorch build ({exc}). "
                "Falling back to CPU. For RTX 5090:\n"
                "  pip install --pre torch --index-url "
                "https://download.pytorch.org/whl/nightly/cu128 --no-cache-dir"
            )
            return "cpu"
    except ImportError:
        return "cpu"


@dataclass
class ExperimentConfig:
    """Single source of truth for Dandelion-Learn simulation experiments."""

    # Cluster
    num_cpu_units: int = 4
    num_gpu_units: int = 2
    num_fpga_units: int = 1
    cpu_speedup: float = 1.0
    gpu_speedup: float = 5.0
    fpga_speedup: float = 3.0
    cpu_energy_per_ms: float = 1.0
    gpu_energy_per_ms: float = 3.0
    fpga_energy_per_ms: float = 2.0
    cpu_cost_per_ms: float = 1.0
    gpu_cost_per_ms: float = 2.5
    fpga_cost_per_ms: float = 2.0

    # Workloads — burstier arrivals so scheduling quality matters
    azure_trace_days: List[int] = field(default_factory=lambda: [1, 2, 3])
    max_jobs_per_day: int = 2000
    mean_interarrival_s: float = 0.012  # was ~0.05; burstier for outstanding gaps
    synthetic_workloads: List[str] = field(
        default_factory=lambda: ["chain", "diamond", "fork_join", "random"]
    )
    jobs_per_synthetic_workload: int = 2000

    # Cold-start penalty model (ms) when a function is not warm on the chosen unit.
    enable_cold_start: bool = True
    cold_start_ms: float = 18.0  # extra service time when func not warm on unit

    # Training
    num_training_episodes: int = 2000
    jobs_per_episode: int = 100
    gnn_epochs: int = 50
    gnn_batch_size: int = 256
    gnn_lr: float = 1e-3
    gnn_hidden: int = 64
    gnn_layers: int = 3
    gnn_weight_decay: float = 1e-5
    seed: int = 42

    # RL (queue-aware contextual bandit)
    rl_epsilon: float = 0.1
    rl_learning_rate: float = 0.01
    rl_context_weight: float = 1.0
    rl_queue_weight: float = 0.8
    rl_cold_weight: float = 0.5

    # Multi-objective optimizer
    optimizer_alpha: float = 1.0
    optimizer_beta: float = 0.05
    optimizer_gamma: float = 0.05
    optimizer_rl_bias: float = 0.5
    optimizer_queue_weight: float = 0.6
    optimizer_cold_weight: float = 0.4

    # Evaluation
    num_runs_per_config: int = 5
    slo_target_ms: float = 100.0
    scalability_sizes: Tuple[int, ...] = (100, 500, 1000, 2000, 5000)

    device: str = "auto"


DEFAULT_CONFIG = ExperimentConfig()
