"""
Evaluation script for distributed multi-node scheduling performance.

Evaluates Dandelion-Learn with distributed coordination across 10-20 nodes
and measures coordination overhead.
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
    InvocationJob,
    HardwareUnit,
    MultiObjectiveOptimizer,
)
from dandelion_learn.gnn_predictor import GNNPredictor
from dandelion_learn.distributed_coordinator import ParameterServer, DistributedRLScheduler
from dandelion_learn.baseline_schedulers import FIFOScheduler


def run_distributed_simulation(
    jobs: List[InvocationJob],
    num_nodes: int,
    hw_units_per_node: int = 7,
) -> Dict:
    """
    Run distributed simulation with multiple nodes.
    
    Args:
        jobs: List of jobs to schedule
        num_nodes: Number of distributed nodes
        hw_units_per_node: Number of hardware units per node
    
    Returns:
        Dictionary with performance metrics and coordination overhead
    """
    # Setup parameter server
    total_hw_units = num_nodes * hw_units_per_node
    param_server = ParameterServer(num_hw_units=total_hw_units, sync_interval=0.1)
    
    # Split jobs round-robin across nodes
    jobs_per_node = len(jobs) // num_nodes
    node_jobs = [
        jobs[i * jobs_per_node:(i + 1) * jobs_per_node]
        for i in range(num_nodes)
    ]
    
    # Shared predictor across nodes
    predictor = GNNPredictor()
    
    # Setup optimizer
    optimizer = MultiObjectiveOptimizer(alpha=1.0, beta=0.05, gamma=0.05)
    
    # Simulate each node
    all_latencies = []
    total_energy = 0.0
    total_cost = 0.0
    
    for node_id in range(num_nodes):
        # Setup hardware for this node
        node_hw_units = build_default_cluster()
        for hw in node_hw_units:
            hw.hw_id = node_id * hw_units_per_node + hw.hw_id
        
        # Create distributed scheduler
        scheduler = DistributedRLScheduler(
            hw_units=node_hw_units,
            parameter_server=param_server,
            node_id=node_id,
            epsilon=0.1,
            sync_interval=0.1
        )
        
        # Setup simulator
        simulator = ClusterSimulator(
            jobs=node_jobs[node_id],
            hw_units=node_hw_units,
            predictor=predictor,
            optimizer=optimizer,
            scheduler=scheduler,
            use_optimizer=True
        )
        
        # Run sim
        simulator.run()
        
        # Collect results
        all_latencies.extend(simulator.job_latencies)
        total_energy += simulator.total_energy
        total_cost += simulator.total_cost
    
    # Get overhead stats
    overhead = param_server.get_coordination_overhead()
    
    # Compute metrics
    latencies_ms = [l * 1000 for l in all_latencies]
    
    return {
        'num_nodes': num_nodes,
        'p50_latency_ms': np.percentile(latencies_ms, 50),
        'p95_latency_ms': np.percentile(latencies_ms, 95),
        'p99_latency_ms': np.percentile(latencies_ms, 99),
        'mean_latency_ms': np.mean(latencies_ms),
        'throughput': len(jobs) / max(all_latencies) if all_latencies else 0,
        'total_energy': total_energy,
        'total_cost': total_cost,
        'coordination_overhead_ms': overhead['avg_sync_time_ms'],
        'total_sync_operations': overhead['total_sync_operations'],
        'conflict_rate_percent': overhead['conflict_rate_percent'],
        'total_overhead_time_ms': overhead['total_overhead_time_ms'],
    }


def evaluate_distributed_scalability():
    """Evaluate distributed scheduling scalability from 1 to 20 nodes."""
    trace_dir = Path("azure_trace")
    jobs = load_azure_invocations(trace_dir, day=1, max_jobs=2000)
    
    results = []
    for num_nodes in [1, 5, 10, 15, 20]:
        print(f"Evaluating {num_nodes} nodes...")
        metrics = run_distributed_simulation(jobs, num_nodes=num_nodes)
        results.append(metrics)
    
    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv("results/distributed_scalability.csv", index=False)
    print(f"\nDistributed scalability results saved to results/distributed_scalability.csv")
    print(df.to_string())
    
    return df


if __name__ == "__main__":
    evaluate_distributed_scalability()

