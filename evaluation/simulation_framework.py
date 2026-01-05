"""
Simulation Framework and Methodology

This module implements the simulation framework:
- Discrete-event simulator (SimPy-based)
- Workload datasets
- Baseline schedulers
- Evaluation metrics
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd
import simpy

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dandelion_learn.dandelion_learn_sim import (
    InvocationJob,
    HardwareUnit,
    load_azure_invocations,
    build_default_cluster,
)
from dandelion_learn.gnn_predictor import GNNPredictor
from dandelion_learn.baseline_schedulers import (
    BaseScheduler,
    FIFOScheduler,
    RandomScheduler,
    RoundRobinScheduler,
    LocalityAwareScheduler,
    ShortestJobFirstScheduler,
)
from dandelion_learn.dandelion_learn_sim import RLScheduler, MultiObjectiveOptimizer, ClusterSimulator


# ============================================================================
# Discrete-event simulator (SimPy-based)
# ============================================================================

# Use ClusterSimulator from dandelion_learn_sim as the discrete-event simulator
DiscreteEventSimulator = ClusterSimulator


# ============================================================================
# Workload datasets
# ============================================================================

@dataclass
class WorkloadDataset:
    """Represents a workload dataset for evaluation."""
    name: str
    jobs: List[InvocationJob]
    source: str  # "azure_trace" or "synthetic_dag"
    description: str


def load_azure_trace_dataset(trace_dir: Path, day: int = 1, max_jobs: int = 2000) -> WorkloadDataset:
    """
    Load Azure Functions trace dataset.
    
    The Azure trace contains real-world serverless invocation patterns:
    - Function identifiers
    - Invocation counts per time period
    - Temporal patterns (bursts, diurnal cycles)
    
    Args:
        trace_dir: Directory containing Azure trace CSV files
        day: Day number (1-14) to load
        max_jobs: Maximum number of jobs to generate from trace
        
    Returns:
        WorkloadDataset with jobs derived from Azure trace
    """
    jobs = load_azure_invocations(trace_dir, day=day, max_jobs=max_jobs)
    return WorkloadDataset(
        name=f"azure_trace_day_{day:02d}",
        jobs=jobs,
        source="azure_trace",
        description=f"Azure Functions trace from day {day} with {len(jobs)} jobs"
    )


def generate_synthetic_dag_workload(
    num_functions: int = 10,
    num_jobs: int = 100,
    dag_structure: str = "chain"
) -> WorkloadDataset:
    """
    Generate synthetic DAG workload for controlled experiments.
    
    DAG structures:
    - "chain": Linear chain of functions
    - "diamond": Diamond-shaped DAG
    - "fork_join": Fork-join parallelism
    - "random": Random DAG structure
    
    Args:
        num_functions: Number of unique function types
        num_jobs: Number of job instances
        dag_structure: Type of DAG structure
        
    Returns:
        WorkloadDataset with synthetic jobs
    """
    jobs: List[InvocationJob] = []
    job_id = 0
    time_cursor = 0.0
    
    for i in range(num_jobs):
        # Select function type based on DAG structure
        if dag_structure == "chain":
            func_id = f"func_{i % num_functions}"
        elif dag_structure == "diamond":
            func_id = f"func_{(i // 3) % num_functions}"
        elif dag_structure == "fork_join":
            func_id = f"func_{i % num_functions}"
        else:  # random
            func_id = f"func_{np.random.randint(0, num_functions)}"
        
        # Synthetic job properties
        base_duration = float(5.0 + np.random.exponential(scale=10.0))
        input_size = float(1.0 + np.random.exponential(scale=5.0))
        
        jobs.append(InvocationJob(
            job_id=job_id,
            func_id=func_id,
            arrival_time=time_cursor,
            input_size=input_size,
            base_duration=base_duration,
        ))
        job_id += 1
        time_cursor += np.random.exponential(scale=0.1)
    
    return WorkloadDataset(
        name=f"synthetic_{dag_structure}",
        jobs=jobs,
        source="synthetic_dag",
        description=f"Synthetic {dag_structure} DAG with {num_functions} functions and {num_jobs} jobs"
    )


# ============================================================================
# Baseline schedulers
# ============================================================================

def get_all_baseline_schedulers() -> Dict[str, Any]:
    """
    Return factory functions for all baseline schedulers.
    
    Baseline schedulers implemented:
    1. FIFO: First-In-First-Out queue
    2. Random: Random hardware selection
    3. Round-Robin: Cyclic assignment across hardware units
    4. Locality-Aware: Prefers hardware with cached data
    5. Shortest-Job-First: Schedules shortest jobs first
    6. Sinan: Workload-aware with burst detection
    7. Fifer: Cost-aware scheduling
    8. X-FaaS: Cross-function optimization with DAG awareness
    9. FIRM: Hybrid heuristic with fairness and resource management
    """
    from dandelion_learn.baseline_schedulers import (
        SinanScheduler, FiferScheduler, XFaaSScheduler, FIRMScheduler
    )
    
    def fifo_factory(hw_units, predictor):
        return FIFOScheduler(hw_units)
    
    def random_factory(hw_units, predictor):
        return RandomScheduler(hw_units)
    
    def round_robin_factory(hw_units, predictor):
        return RoundRobinScheduler(hw_units)
    
    def locality_aware_factory(hw_units, predictor):
        return LocalityAwareScheduler(hw_units)
    
    def sjf_factory(hw_units, predictor):
        return ShortestJobFirstScheduler(hw_units, predictor)
    
    def sinan_factory(hw_units, predictor):
        return SinanScheduler(hw_units)
    
    def fifer_factory(hw_units, predictor):
        return FiferScheduler(hw_units, predictor)
    
    def xfaas_factory(hw_units, predictor):
        return XFaaSScheduler(hw_units, predictor)
    
    def firm_factory(hw_units, predictor):
        return FIRMScheduler(hw_units, predictor)
    
    return {
        "FIFO": fifo_factory,
        "Random": random_factory,
        "Round-Robin": round_robin_factory,
        "Locality-Aware": locality_aware_factory,
        "Shortest-Job-First": sjf_factory,
        "Sinan": sinan_factory,
        "Fifer": fifer_factory,
        "X-FaaS": xfaas_factory,
        "FIRM": firm_factory,
    }


# ============================================================================
# Evaluation metrics
# ============================================================================

@dataclass
class EvaluationMetrics:
    """
    Comprehensive evaluation metrics.
    
    Metrics include:
    - Latency metrics (p50, p95, p99, mean, std)
    - Throughput (jobs per second)
    - Resource utilization (per hardware unit)
    - Energy consumption
    - Cost
    - SLO compliance
    """
    scheduler_name: str
    workload_name: str
    
    # Latency metrics (milliseconds)
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    mean_latency_ms: float
    std_latency_ms: float
    
    # Throughput
    throughput_jobs_per_sec: float
    
    # Resource utilization (percentage)
    hw_utilization: Dict[int, float]
    avg_gpu_utilization: float
    
    # Energy and cost
    total_energy: float
    total_cost: float
    energy_per_job: float
    cost_per_job: float
    
    # SLO compliance
    slo_violations: int
    slo_violation_rate: float
    
    # Simulation metadata
    num_jobs: int
    simulation_time_s: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV export."""
        return {
            'scheduler': self.scheduler_name,
            'workload': self.workload_name,
            'p50_latency_ms': self.p50_latency_ms,
            'p95_latency_ms': self.p95_latency_ms,
            'p99_latency_ms': self.p99_latency_ms,
            'mean_latency_ms': self.mean_latency_ms,
            'std_latency_ms': self.std_latency_ms,
            'throughput': self.throughput_jobs_per_sec,
            'total_energy': self.total_energy,
            'total_cost': self.total_cost,
            'energy_per_job': self.energy_per_job,
            'cost_per_job': self.cost_per_job,
            'slo_violations': self.slo_violations,
            'slo_violation_rate': self.slo_violation_rate,
            'num_jobs': self.num_jobs,
            'simulation_time_s': self.simulation_time_s,
            **{f'hw_{hw_id}_util': util for hw_id, util in self.hw_utilization.items()},
        }


def compute_metrics_from_simulator(
    sim: ClusterSimulator,
    scheduler_name: str,
    workload_name: str
) -> EvaluationMetrics:
    """
    Extract evaluation metrics from simulator results.
    
    Args:
        sim: Completed simulator instance
        scheduler_name: Name of the scheduler used
        workload_name: Name of the workload used
        
    Returns:
        EvaluationMetrics with all computed metrics
    """
    if not sim.job_latencies:
        # Return empty metrics if no jobs completed
        return EvaluationMetrics(
            scheduler_name=scheduler_name,
            workload_name=workload_name,
            p50_latency_ms=0.0, p95_latency_ms=0.0, p99_latency_ms=0.0,
            mean_latency_ms=0.0, std_latency_ms=0.0,
            throughput_jobs_per_sec=0.0,
            hw_utilization={}, avg_gpu_utilization=0.0,
            total_energy=0.0, total_cost=0.0,
            energy_per_job=0.0, cost_per_job=0.0,
            slo_violations=0, slo_violation_rate=0.0,
            num_jobs=0, simulation_time_s=0.0,
        )
    
    latencies_ms = np.array(sim.job_latencies) * 1000.0
    
    # Calculate total simulation time (max completion time)
    total_time = max(sim.job_latencies) if sim.job_latencies else 0.0
    
    # Calculate throughput
    throughput = len(sim.jobs) / total_time if total_time > 0 else 0.0
    
    # Calculate hardware utilization percentages
    hw_util_pct = {}
    for hw_id, busy_time in sim.hw_utilization.items():
        util_pct = (busy_time / total_time * 100.0) if total_time > 0 else 0.0
        hw_util_pct[hw_id] = util_pct
    
    # Extract GPU utilization (assuming hw_id 1+ are GPU)
    gpu_util = 0.0
    for hw in sim.hw_units:
        if hw.hw_type == "gpu":
            gpu_util = max(gpu_util, hw_util_pct.get(hw.hw_id, 0.0))
    
    # Calculate per-job metrics
    num_jobs = len(sim.jobs)
    energy_per_job = sim.total_energy / num_jobs if num_jobs > 0 else 0.0
    cost_per_job = sim.total_cost / num_jobs if num_jobs > 0 else 0.0
    
    # Count SLO violations (assuming 100ms SLO)
    slo_target_ms = 100.0
    slo_violations = np.sum(latencies_ms > slo_target_ms)
    slo_violation_rate = slo_violations / num_jobs if num_jobs > 0 else 0.0
    
    return EvaluationMetrics(
        scheduler_name=scheduler_name,
        workload_name=workload_name,
        p50_latency_ms=float(np.percentile(latencies_ms, 50)),
        p95_latency_ms=float(np.percentile(latencies_ms, 95)),
        p99_latency_ms=float(np.percentile(latencies_ms, 99)),
        mean_latency_ms=float(np.mean(latencies_ms)),
        std_latency_ms=float(np.std(latencies_ms)),
        throughput_jobs_per_sec=throughput,
        hw_utilization=hw_util_pct,
        avg_gpu_utilization=gpu_util,
        total_energy=sim.total_energy,
        total_cost=sim.total_cost,
        energy_per_job=energy_per_job,
        cost_per_job=cost_per_job,
        slo_violations=int(slo_violations),
        slo_violation_rate=slo_violation_rate,
        num_jobs=num_jobs,
        simulation_time_s=total_time,
    )


# ============================================================================
# Main execution for testing
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SIMULATION FRAMEWORK AND METHODOLOGY")
    print("=" * 70)
    print()
    
    # Test Discrete-event simulator
    print("Testing Discrete-event simulator...")
    root = Path(".").resolve()
    trace_dir = root / "azure_trace"
    
    if trace_dir.exists():
        dataset = load_azure_trace_dataset(trace_dir, day=1, max_jobs=100)
        hw_units = build_default_cluster()
        predictor = GNNPredictor()
        predictor.fit(dataset.jobs)
        
        scheduler = FIFOScheduler(hw_units)
        sim = ClusterSimulator(
            jobs=dataset.jobs[:50],
            hw_units=hw_units,
            predictor=predictor,
            scheduler=scheduler,
            optimizer=None,
            use_optimizer=False,
        )
        sim.run()
        metrics = compute_metrics_from_simulator(sim, "FIFO", dataset.name)
        print(f"  Simulated {metrics.num_jobs} jobs")
        print(f"  p99 latency: {metrics.p99_latency_ms:.2f} ms")
        print(f"  Throughput: {metrics.throughput_jobs_per_sec:.2f} jobs/s")
        print("[OK] Discrete-event simulator working")
    else:
        print("[WARNING] Azure trace directory not found, skipping test")
    
    print()
    print("Workload datasets available:")
    print("  - Azure trace datasets (real-world)")
    print("  - Synthetic DAG workloads (chain, diamond, fork_join, random)")
    print()
    
    print("Baseline schedulers available:")
    baselines = get_all_baseline_schedulers()
    for name in baselines.keys():
        print(f"  - {name}")
    print()
    
    print("Evaluation metrics defined:")
    print("  - Latency (p50, p95, p99, mean, std)")
    print("  - Throughput")
    print("  - Resource utilization")
    print("  - Energy and cost")
    print("  - SLO compliance")
    print()
    print("[OK] Simulation framework ready")

