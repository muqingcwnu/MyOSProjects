"""
Statistical significance tests for latency and throughput results.

Performs t-tests to compare Dandelion-Learn against baseline schedulers.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from scipy import stats


def run_multiple_runs(scheduler_name: str, num_runs: int = 10) -> Dict[str, List[float]]:
    """
    Run multiple independent runs of a scheduler.
    
    Returns:
        Dictionary with lists of metrics across runs
    """
    from dandelion_learn.dandelion_learn_sim import (
        ClusterSimulator,
        build_default_cluster,
        load_azure_invocations,
        RLScheduler,
        MultiObjectiveOptimizer,
    )
    from dandelion_learn.gnn_predictor import GNNPredictor
    from dandelion_learn.baseline_schedulers import (
        FIFOScheduler,
        RandomScheduler,
        RoundRobinScheduler,
        LocalityAwareScheduler,
        ShortestJobFirstScheduler,
        SinanScheduler,
        FiferScheduler,
        XFaaSScheduler,
        FIRMScheduler,
    )
    
    trace_dir = Path("azure_trace")
    jobs = load_azure_invocations(trace_dir, day=1, max_jobs=1000)
    
    hw_units = build_default_cluster()
    predictor = GNNPredictor()
    optimizer = MultiObjectiveOptimizer(alpha=1.0, beta=0.05, gamma=0.05)
    
    # Create scheduler based on name
    if scheduler_name == "Dandelion-Learn":
        scheduler = RLScheduler(hw_units=hw_units, epsilon=0.1)
        use_optimizer = True
    elif scheduler_name == "FIFO":
        scheduler = FIFOScheduler(hw_units=hw_units)
        use_optimizer = False
    elif scheduler_name == "Random":
        scheduler = RandomScheduler(hw_units=hw_units)
        use_optimizer = False
    elif scheduler_name == "Round-Robin":
        scheduler = RoundRobinScheduler(hw_units=hw_units)
        use_optimizer = False
    elif scheduler_name == "Locality-Aware":
        scheduler = LocalityAwareScheduler(hw_units=hw_units)
        use_optimizer = False
    elif scheduler_name == "Shortest-Job-First":
        scheduler = ShortestJobFirstScheduler(hw_units=hw_units, predictor=predictor)
        use_optimizer = False
    elif scheduler_name == "Sinan":
        scheduler = SinanScheduler(hw_units=hw_units)
        use_optimizer = False
    elif scheduler_name == "Fifer":
        scheduler = FiferScheduler(hw_units=hw_units, predictor=predictor)
        use_optimizer = False
    elif scheduler_name == "X-FaaS":
        scheduler = XFaaSScheduler(hw_units=hw_units, predictor=predictor)
        use_optimizer = False
    elif scheduler_name == "FIRM":
        scheduler = FIRMScheduler(hw_units=hw_units, predictor=predictor)
        use_optimizer = False
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    p99_latencies = []
    throughputs = []
    
    for run_id in range(num_runs):
        simulator = ClusterSimulator(
            jobs=jobs,
            hw_units=hw_units,
            predictor=predictor,
            optimizer=optimizer,
            scheduler=scheduler,
            use_optimizer=use_optimizer
        )
        
        simulator.run()
        
        latencies_ms = [l * 1000 for l in simulator.job_latencies]
        p99_latencies.append(np.percentile(latencies_ms, 99))
        throughputs.append(len(jobs) / max(simulator.job_latencies) if simulator.job_latencies else 0)
    
    return {
        'p99_latency': p99_latencies,
        'throughput': throughputs
    }


def perform_statistical_tests():
    """Perform t-tests comparing Dandelion-Learn against all baselines."""
    schedulers = [
        "Dandelion-Learn",
        "FIFO",
        "Random",
        "Round-Robin",
        "Locality-Aware",
        "Shortest-Job-First",
        "Sinan",
        "Fifer",
        "X-FaaS",
        "FIRM",
    ]
    
    print("Running multiple runs for statistical significance...")
    all_results = {}
    for scheduler in schedulers:
        print(f"  Running {scheduler}...")
        all_results[scheduler] = run_multiple_runs(scheduler, num_runs=10)
    
    # Perform t-tests
    dl_p99 = all_results["Dandelion-Learn"]['p99_latency']
    dl_throughput = all_results["Dandelion-Learn"]['throughput']
    
    test_results = []
    
    for scheduler in schedulers:
        if scheduler == "Dandelion-Learn":
            continue
        
        baseline_p99 = all_results[scheduler]['p99_latency']
        baseline_throughput = all_results[scheduler]['throughput']
        
        # T-test for p99 latency (one-tailed: Dandelion-Learn should be lower)
        t_stat_latency, p_value_latency = stats.ttest_ind(dl_p99, baseline_p99, alternative='less')
        
        # T-test for throughput (one-tailed: Dandelion-Learn should be higher)
        t_stat_throughput, p_value_throughput = stats.ttest_ind(dl_throughput, baseline_throughput, alternative='greater')
        
        test_results.append({
            'baseline': scheduler,
            'dl_p99_mean': np.mean(dl_p99),
            'baseline_p99_mean': np.mean(baseline_p99),
            'p99_improvement_percent': (np.mean(baseline_p99) - np.mean(dl_p99)) / np.mean(baseline_p99) * 100,
            'p99_t_statistic': t_stat_latency,
            'p99_p_value': p_value_latency,
            'p99_significant': p_value_latency < 0.05,
            'dl_throughput_mean': np.mean(dl_throughput),
            'baseline_throughput_mean': np.mean(baseline_throughput),
            'throughput_improvement_percent': (np.mean(dl_throughput) - np.mean(baseline_throughput)) / np.mean(baseline_throughput) * 100,
            'throughput_t_statistic': t_stat_throughput,
            'throughput_p_value': p_value_throughput,
            'throughput_significant': p_value_throughput < 0.05,
        })
    
    df = pd.DataFrame(test_results)
    df.to_csv("results/statistical_significance_tests.csv", index=False)
    
    print("\nStatistical significance test results:")
    print(df.to_string())
    
    return df


if __name__ == "__main__":
    perform_statistical_tests()

