"""
Sensitivity analysis for optimizer weights and RL exploration rate.

Evaluates Dandelion-Learn performance across different parameter configurations.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from typing import Dict, List
import pandas as pd
import numpy as np

from dandelion_learn.dandelion_learn_sim import (
    ClusterSimulator,
    build_default_cluster,
    load_azure_invocations,
    RLScheduler,
    MultiObjectiveOptimizer,
)
from dandelion_learn.gnn_predictor import GNNPredictor


def run_sensitivity_experiment(
    jobs: List,
    alpha: float,
    beta: float,
    gamma: float,
    epsilon: float,
) -> Dict:
    """Run single experiment with given parameters."""
    hw_units = build_default_cluster()
    predictor = GNNPredictor()
    optimizer = MultiObjectiveOptimizer(alpha=alpha, beta=beta, gamma=gamma)
    scheduler = RLScheduler(hw_units=hw_units, epsilon=epsilon)
    
    simulator = ClusterSimulator(
        jobs=jobs,
        hw_units=hw_units,
        predictor=predictor,
        optimizer=optimizer,
        scheduler=scheduler,
        use_optimizer=True
    )
    
    simulator.run()
    
    latencies_ms = [l * 1000 for l in simulator.job_latencies]
    
    return {
        'alpha': alpha,
        'beta': beta,
        'gamma': gamma,
        'epsilon': epsilon,
        'p50_latency_ms': np.percentile(latencies_ms, 50),
        'p95_latency_ms': np.percentile(latencies_ms, 95),
        'p99_latency_ms': np.percentile(latencies_ms, 99),
        'mean_latency_ms': np.mean(latencies_ms),
        'throughput': len(jobs) / max(simulator.job_latencies) if simulator.job_latencies else 0,
        'total_energy': simulator.total_energy,
        'total_cost': simulator.total_cost,
    }


def evaluate_optimizer_weights():
    """Evaluate sensitivity to optimizer weights (α, β, γ)."""
    from dandelion_learn.paths import get_azure_trace_dir
    jobs = load_azure_invocations(get_azure_trace_dir(), day=1, max_jobs=1000)
    
    results = []
    
    # Baseline: α=1.0, β=0.05, γ=0.05
    # Vary each weight independently
    weight_configs = [
        # Vary alpha (latency weight)
        (0.5, 0.05, 0.05),
        (0.8, 0.05, 0.05),
        (1.0, 0.05, 0.05),  # baseline
        (1.2, 0.05, 0.05),
        (1.5, 0.05, 0.05),
        # Vary beta (energy weight)
        (1.0, 0.01, 0.05),
        (1.0, 0.03, 0.05),
        (1.0, 0.05, 0.05),  # baseline
        (1.0, 0.10, 0.05),
        (1.0, 0.15, 0.05),
        # Vary gamma (cost weight)
        (1.0, 0.05, 0.01),
        (1.0, 0.05, 0.03),
        (1.0, 0.05, 0.05),  # baseline
        (1.0, 0.05, 0.10),
        (1.0, 0.05, 0.15),
    ]
    
    for alpha, beta, gamma in weight_configs:
        print(f"Testing α={alpha}, β={beta}, γ={gamma}...")
        metrics = run_sensitivity_experiment(jobs, alpha, beta, gamma, epsilon=0.1)
        results.append(metrics)
    
    df = pd.DataFrame(results)
    out = Path(__file__).resolve().parents[1] / "results_fixed" / "optimizer_weight_sensitivity.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nOptimizer weight sensitivity results saved")
    return df


def evaluate_rl_exploration_rate():
    """Evaluate sensitivity to RL exploration rate (ε)."""
    from dandelion_learn.paths import get_azure_trace_dir
    jobs = load_azure_invocations(get_azure_trace_dir(), day=1, max_jobs=1000)
    
    results = []
    
    # Vary epsilon from 0.0 (pure exploitation) to 0.5 (high exploration)
    epsilon_values = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    for epsilon in epsilon_values:
        print(f"Testing ε={epsilon}...")
        metrics = run_sensitivity_experiment(jobs, alpha=1.0, beta=0.05, gamma=0.05, epsilon=epsilon)
        results.append(metrics)
    
    df = pd.DataFrame(results)
    out = Path(__file__).resolve().parents[1] / "results_fixed" / "rl_exploration_sensitivity.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nRL exploration rate sensitivity results saved")
    return df


if __name__ == "__main__":
    print("Running optimizer weight sensitivity analysis...")
    evaluate_optimizer_weights()
    
    print("\nRunning RL exploration rate sensitivity analysis...")
    evaluate_rl_exploration_rate()

