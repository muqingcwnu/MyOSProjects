from __future__ import annotations

"""
Simplified Dandelion-Learn style scheduler over a serverless workload.

This file is a lightweight implementation that keeps the core structure
of Dandelion-Learn while staying runnable and using the Azure trace as
the workload source.

Key components:
  - GNNPredictor: implemented in `gnn_predictor.py` with message-passing
    and prediction equations.
  - RLScheduler: an online bandit-style agent deciding placement.
  - MultiObjectiveOptimizer: combines latency / energy / cost into a score.
  - SimPy-based discrete-event simulation of workers and function invocations.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import simpy

from dandelion_learn.gnn_predictor import GNNPredictor, Prediction


# -----------------------------
# Workload: Azure trace wrapper
# -----------------------------


@dataclass
class InvocationJob:
    """Single function invocation in the simulator."""

    job_id: int
    func_id: str
    arrival_time: float  # in seconds
    input_size: float  # arbitrary units (e.g., KB)
    base_duration: float  # baseline runtime estimate (ms)


def load_azure_invocations(trace_dir: Path, day: int = 1, max_jobs: int = 2000) -> List[InvocationJob]:
    """
    Load a small subset of Azure invocations as jobs.

    We do not depend strongly on the exact schema; we try to infer:
      - function identifier column (HashFunction / FunctionId / FunctionName)
      - count column (Total / Count / NumInvocations)
    and then expand counts into individual jobs with synthetic arrival times.
    """

    inv_path = trace_dir / f"invocations_per_function_md.anon.d{day:02d}.csv"
    if not inv_path.exists():
        raise FileNotFoundError(f"Missing invocation trace file: {inv_path}")

    df = pd.read_csv(inv_path)

    func_col_candidates = ["HashFunction", "FunctionId", "FunctionName"]
    func_col = next((c for c in func_col_candidates if c in df.columns), None)
    if func_col is None:
        raise ValueError(
            f"Could not find function column in {inv_path}. "
            f"Have columns={list(df.columns)[:10]}..."
        )

    # Numbered columns are invocations per hour, sum them up
    numeric_cols = [c for c in df.columns if c.isdigit() or (isinstance(c, str) and c.replace('.', '').isdigit())]
    if not numeric_cols:
        # Try Total/Count columns as fallback
        count_col_candidates = ["Total", "Count", "NumInvocations"]
        count_col = next((c for c in count_col_candidates if c in df.columns), None)
        if count_col is None:
            raise ValueError(f"No numeric columns found in {inv_path}")
        numeric_cols = [count_col]

    # Convert invocations to jobs, subsample to keep it manageable
    jobs: List[InvocationJob] = []
    job_id = 0
    time_cursor = 0.0
    for _, row in df.iterrows():
        func_id = str(row[func_col])
        # Sum up all invocations
        count = int(sum(float(row[c]) for c in numeric_cols if pd.notna(row[c])))
        if count <= 0:
            continue

        # Base duration scales with log of count
        base_duration = float(5.0 + 0.5 * np.log1p(count))  # ms
        input_size = float(1.0 + np.log1p(count))

        # Spread jobs to create bursts, cap at 10 per function
        num_jobs_for_func = min(count, 10)
        for _ in range(num_jobs_for_func):
            jobs.append(
                InvocationJob(
                    job_id=job_id,
                    func_id=func_id,
                    arrival_time=time_cursor,
                    input_size=input_size,
                    base_duration=base_duration,
                )
            )
            job_id += 1
            time_cursor += np.random.exponential(scale=0.05)  # 50ms average inter-arrival
            if job_id >= max_jobs:
                break
        if job_id >= max_jobs:
            break

    return jobs


# -----------------------------
# Multi-objective optimizer
# -----------------------------


@dataclass
class HardwareUnit:
    hw_id: int
    hw_type: str  # "cpu" or "gpu"
    speedup: float  # relative to baseline runtime
    energy_per_ms: float
    cost_per_ms: float


class MultiObjectiveOptimizer:
    """
    Combines latency, energy, and cost into a single scalar score.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.1, gamma: float = 0.1) -> None:
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def score_assignment(
        self, job: InvocationJob, pred: Prediction, hw: HardwareUnit
    ) -> float:
        runtime = pred.runtime_ms / hw.speedup
        energy = runtime * hw.energy_per_ms
        cost = runtime * hw.cost_per_ms
        return self.alpha * runtime + self.beta * energy + self.gamma * cost


# -----------------------------
# RL-style scheduler (bandit)
# -----------------------------


class RLScheduler:
    """
    Simplified RL agent:
      - Treat each hardware unit as an arm in a multi-armed bandit.
      - Reward = negative end-to-end completion time per job.
      - Uses epsilon-greedy exploration.
    """

    def __init__(self, hw_units: List[HardwareUnit], epsilon: float = 0.1) -> None:
        self.hw_units = hw_units
        self.epsilon = epsilon
        self.counts = np.zeros(len(hw_units), dtype=np.int64)
        # Small random init to break symmetry
        self.values = np.random.uniform(-0.1, 0.1, size=len(hw_units)).astype(np.float64)

    def select_hw(self, job: InvocationJob, now: float) -> HardwareUnit:
        if np.random.rand() < self.epsilon:
            idx = np.random.randint(len(self.hw_units))
        else:
            idx = int(np.argmax(self.values))
        return self.hw_units[idx]

    def update(self, hw: HardwareUnit, job: InvocationJob, completion_time: float, arrival_time: float) -> None:
        latency = completion_time - arrival_time
        reward = -latency
        idx = hw.hw_id
        self.counts[idx] += 1
        n = self.counts[idx]
        value = self.values[idx]
        self.values[idx] = value + (reward - value) / n


# -----------------------------
# SimPy-based worker simulation
# -----------------------------


class ClusterSimulator:
    """
    Discrete-event simulator for serverless invocations scheduled by
    a learned-style scheduler or baseline heuristic scheduler.
    """

    def __init__(
        self,
        jobs: List[InvocationJob],
        hw_units: List[HardwareUnit],
        predictor: GNNPredictor,
        optimizer: Optional[MultiObjectiveOptimizer],
        scheduler,  # Accept any scheduler with select_hw() and update() methods
        use_optimizer: bool = True,
    ) -> None:
        self.env = simpy.Environment()
        self.jobs = sorted(jobs, key=lambda j: j.arrival_time)
        self.hw_units = hw_units
        self.predictor = predictor
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.use_optimizer = use_optimizer and optimizer is not None
        self.resources: Dict[int, simpy.Resource] = {
            hw.hw_id: simpy.Resource(self.env, capacity=1) for hw in hw_units
        }
        self.job_latencies: List[float] = []
        self.hw_utilization: Dict[int, float] = {hw.hw_id: 0.0 for hw in hw_units}
        self.total_energy: float = 0.0
        self.total_cost: float = 0.0

    def run(self) -> None:
        self.env.process(self._job_arrival_process())
        self.env.run()

    def _job_arrival_process(self):
        for job in self.jobs:
            yield self.env.timeout(max(0.0, job.arrival_time - self.env.now))
            self.env.process(self._handle_job(job))

    def _handle_job(self, job: InvocationJob):
        # Get predictions for all hardware types
        preds = {hw.hw_id: self.predictor.predict(job, hw_type=hw.hw_type) for hw in self.hw_units}

        # Scheduler picks a candidate
        candidate_hw = self.scheduler.select_hw(job, now=self.env.now)

        # If optimizer enabled, evaluate all options and pick best
        if self.use_optimizer:
            best_hw = candidate_hw
            best_score = self.optimizer.score_assignment(job, preds[candidate_hw.hw_id], candidate_hw)
            for hw in self.hw_units:
                score = self.optimizer.score_assignment(job, preds[hw.hw_id], hw)
                if score < best_score:
                    best_score = score
                    best_hw = hw
            chosen_hw = best_hw
        else:
            # Baselines use their direct selection
            chosen_hw = candidate_hw

        res = self.resources[chosen_hw.hw_id]
        with res.request() as req:
            yield req
            pred = preds[chosen_hw.hw_id]
            runtime_ms = pred.runtime_ms / chosen_hw.speedup
            runtime_s = max(0.0001, runtime_ms / 1000.0)
            start_time = self.env.now
            yield self.env.timeout(runtime_s)
            completion_time = self.env.now
            latency = completion_time - job.arrival_time
            self.job_latencies.append(latency)
            
            # Track metrics
            self.hw_utilization[chosen_hw.hw_id] += runtime_s
            self.total_energy += runtime_ms * chosen_hw.energy_per_ms
            self.total_cost += runtime_ms * chosen_hw.cost_per_ms
            
            # Update scheduler if it supports learning
            if hasattr(self.scheduler, 'update'):
                self.scheduler.update(chosen_hw, job, completion_time, job.arrival_time)


def build_default_cluster() -> List[HardwareUnit]:
    return [
        HardwareUnit(hw_id=0, hw_type="cpu", speedup=1.0, energy_per_ms=1.0, cost_per_ms=1.0),
        HardwareUnit(hw_id=1, hw_type="cpu", speedup=1.2, energy_per_ms=1.2, cost_per_ms=1.1),
        HardwareUnit(hw_id=2, hw_type="gpu", speedup=3.0, energy_per_ms=2.0, cost_per_ms=2.5),
    ]


def run_dandelion_learn_simulation(root: Path, day: int = 1, max_jobs: int = 2000) -> None:
    """
    End-to-end experiment:
      - Load a subset of Azure invocations.
      - Train the simple predictor.
      - Run the learned-style scheduler in a SimPy simulator.
      - Report basic latency statistics (p50/p95/p99).
    """

    trace_dir = root / "azure_trace"
    jobs = load_azure_invocations(trace_dir, day=day, max_jobs=max_jobs)
    if not jobs:
        raise RuntimeError("No jobs loaded from Azure trace.")

    predictor = GNNPredictor()
    predictor.fit(jobs)

    hw_units = build_default_cluster()
    optimizer = MultiObjectiveOptimizer(alpha=1.0, beta=0.05, gamma=0.05)
    scheduler = RLScheduler(hw_units, epsilon=0.1)

    sim = ClusterSimulator(
        jobs=jobs,
        hw_units=hw_units,
        predictor=predictor,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    sim.run()

    latencies = np.array(sim.job_latencies)
    if latencies.size == 0:
        print("Simulation produced no completed jobs.")
        return

    p50 = np.percentile(latencies, 50) * 1000
    p95 = np.percentile(latencies, 95) * 1000
    p99 = np.percentile(latencies, 99) * 1000

    print(f"Jobs simulated: {len(latencies)}")
    print(f"Latency (ms): p50={p50:.2f}, p95={p95:.2f}, p99={p99:.2f}")


if __name__ == "__main__":
    run_dandelion_learn_simulation(Path(".").resolve())


