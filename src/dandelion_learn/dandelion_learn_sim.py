from __future__ import annotations

"""
Dandelion-Learn style scheduler over a serverless workload (SimPy).

Pipeline:
  1) GNN predicts runtime / memory / affinity (scheduling only).
  2) Contextual bandit RL selects hardware using Q-values + GNN context.
  3) Optional multi-objective optimizer re-ranks with latency/energy/cost,
     biased by RL Q-values.
  4) Execution uses ground-truth base_duration with affinity-aware speedup
     (predictions never drive simulated service time).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import simpy

from dandelion_learn.experiment_config import DEFAULT_CONFIG, ExperimentConfig
from dandelion_learn.gnn_predictor import (
    GNNPredictor,
    Prediction,
    ground_truth_gpu_affinity,
)
from dandelion_learn.paths import get_azure_trace_dir, require_azure_day


# -----------------------------
# Workload: Azure trace wrapper
# -----------------------------


@dataclass
class InvocationJob:
    """Single function invocation in the simulator."""

    job_id: int
    func_id: str
    arrival_time: float  # seconds
    input_size: float
    base_duration: float  # ground-truth baseline runtime (ms)


def load_azure_invocations(
    trace_dir: Optional[Path] = None,
    day: int = 1,
    max_jobs: int = 2000,
) -> List[InvocationJob]:
    """
    Load Azure Functions 2019 invocations from extracted CSVs.

    Default directory: <project>/azure_trace/
    Expected file: invocations_per_function_md.anon.dXX.csv
    Schema: HashOwner, HashApp, HashFunction, Trigger, 1..1440 (per-minute counts)
    """
    resolved = get_azure_trace_dir(trace_dir)
    inv_path = require_azure_day(resolved, day)

    df = pd.read_csv(inv_path)

    # Azure Functions Dataset 2019 uses HashFunction
    func_col_candidates = ["HashFunction", "FunctionId", "FunctionName", "HashApp"]
    func_col = next((c for c in func_col_candidates if c in df.columns), None)
    if func_col is None:
        raise ValueError(
            f"Could not find function column in {inv_path}. "
            f"Have columns={list(df.columns)[:10]}..."
        )

    # Per-minute invocation columns are named "1".."1440"
    numeric_cols = [
        c
        for c in df.columns
        if (isinstance(c, int) or (isinstance(c, str) and c.isdigit()))
    ]
    if not numeric_cols:
        count_col_candidates = ["Total", "Count", "NumInvocations"]
        count_col = next((c for c in count_col_candidates if c in df.columns), None)
        if count_col is None:
            raise ValueError(f"No numeric minute columns found in {inv_path}")
        numeric_cols = [count_col]

    jobs: List[InvocationJob] = []
    job_id = 0
    time_cursor = 0.0
    for _, row in df.iterrows():
        func_id = str(row[func_col])
        count = int(sum(float(row[c]) for c in numeric_cols if pd.notna(row[c])))
        if count <= 0:
            continue

        base_duration = float(5.0 + 0.5 * np.log1p(count))
        input_size = float(1.0 + np.log1p(count))

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
            time_cursor += np.random.exponential(scale=float(DEFAULT_CONFIG.mean_interarrival_s))
            if job_id >= max_jobs:
                break
        if job_id >= max_jobs:
            break

    return jobs


# -----------------------------
# Hardware + multi-objective
# -----------------------------


@dataclass
class HardwareUnit:
    hw_id: int
    hw_type: str  # "cpu", "gpu", "fpga"
    speedup: float
    energy_per_ms: float
    cost_per_ms: float


def predicted_speedup(pred: Prediction, hw: HardwareUnit) -> float:
    """Scheduling-time speedup proxy from GNN affinity (not ground truth)."""
    if hw.hw_type == "cpu":
        return max(0.5, hw.speedup)
    aff = float(np.clip(pred.gpu_affinity, 0.0, 1.0))
    if hw.hw_type == "fpga":
        aff = float(np.clip(0.55 * aff + 0.25, 0.0, 1.0))
    if aff < 0.35:
        return max(0.4, 0.55 + 0.6 * aff)
    return max(0.5, 1.0 + (hw.speedup - 1.0) * ((aff - 0.35) / 0.65) ** 0.85)


def effective_speedup(job: InvocationJob, hw: HardwareUnit) -> float:
    """
    Ground-truth service-time speedup with strong mismatch penalty.
    Wrong accelerator can be slower than CPU — makes learned placement matter.
    """
    if hw.hw_type == "cpu":
        return max(0.5, hw.speedup)

    aff = ground_truth_gpu_affinity(job)
    if hw.hw_type == "fpga":
        aff = float(np.clip(0.55 * aff + 0.2 * (1.0 - abs(aff - 0.55)), 0.0, 1.0))

    if aff < 0.35:
        return max(0.4, 0.55 + 0.6 * aff)
    return max(0.5, 1.0 + (hw.speedup - 1.0) * ((aff - 0.35) / 0.65) ** 0.85)


class MultiObjectiveOptimizer:
    """
    Latency / energy / cost score with optional queue wait + cold-start penalties.
    Lower is better.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.05,
        gamma: float = 0.05,
        queue_weight: float = 0.6,
        cold_weight: float = 0.4,
        cold_start_ms: float = 18.0,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.queue_weight = queue_weight
        self.cold_weight = cold_weight
        self.cold_start_ms = cold_start_ms

    def score_assignment(
        self,
        job: InvocationJob,
        pred: Prediction,
        hw: HardwareUnit,
        queue_len: float = 0.0,
        is_cold: bool = False,
    ) -> float:
        spd = predicted_speedup(pred, hw)
        runtime = pred.runtime_ms / spd
        if is_cold:
            runtime += self.cold_start_ms
        # Approximate wait: each queued job ~ predicted runtime on that unit
        wait = queue_len * max(1.0, runtime)
        energy = runtime * hw.energy_per_ms
        cost = runtime * hw.cost_per_ms
        cold_pen = self.cold_start_ms if is_cold else 0.0
        return (
            self.alpha * runtime
            + self.beta * energy
            + self.gamma * cost
            + self.queue_weight * wait
            + self.cold_weight * cold_pen
        )


# -----------------------------
# Contextual bandit RL
# -----------------------------


class RLScheduler:
    """
    Queue- and cold-start-aware epsilon-greedy bandit.

    score_i = Q_i - w_lat*pred_lat - w_q*queue - w_c*cold + affinity_bonus
    """

    def __init__(
        self,
        hw_units: List[HardwareUnit],
        epsilon: float = 0.1,
        use_context: bool = True,
        context_weight: float = 1.0,
        queue_weight: float = 0.8,
        cold_weight: float = 0.5,
        cold_start_ms: float = 18.0,
    ) -> None:
        self.hw_units = hw_units
        self.epsilon = epsilon
        self.use_context = use_context
        self.context_weight = context_weight
        self.queue_weight = queue_weight
        self.cold_weight = cold_weight
        self.cold_start_ms = cold_start_ms
        self.counts = np.zeros(len(hw_units), dtype=np.int64)
        self.values = np.random.uniform(-0.1, 0.1, size=len(hw_units)).astype(np.float64)
        # Warm cache: hw_id -> last func_id (updated by simulator)
        self.warm_func: Dict[int, Optional[str]] = {hw.hw_id: None for hw in hw_units}

    def select_hw(
        self,
        job: InvocationJob,
        now: float,
        predictions: Optional[Dict[int, Prediction]] = None,
        queue_lengths: Optional[Dict[int, int]] = None,
    ) -> HardwareUnit:
        if np.random.rand() < self.epsilon:
            return self.hw_units[int(np.random.randint(len(self.hw_units)))]

        qlen = queue_lengths or {hw.hw_id: 0 for hw in self.hw_units}

        if self.use_context and predictions is not None:
            scores = []
            for i, hw in enumerate(self.hw_units):
                pred = predictions[hw.hw_id]
                spd = predicted_speedup(pred, hw)
                pred_lat = pred.runtime_ms / spd
                cold = 1.0 if self.warm_func.get(hw.hw_id) != job.func_id else 0.0
                aff_bonus = (
                    0.15 * (1.0 - pred.gpu_affinity)
                    if hw.hw_type == "cpu"
                    else 0.55 * pred.gpu_affinity
                )
                score = (
                    self.values[i]
                    - self.context_weight * 0.02 * pred_lat
                    - self.queue_weight * 0.05 * float(qlen.get(hw.hw_id, 0))
                    - self.cold_weight * 0.02 * cold * self.cold_start_ms
                    + aff_bonus
                )
                scores.append(score)
            return self.hw_units[int(np.argmax(scores))]

        # No context: prefer high Q and low queue
        scores = [
            self.values[i] - self.queue_weight * 0.05 * float(qlen.get(hw.hw_id, 0))
            for i, hw in enumerate(self.hw_units)
        ]
        return self.hw_units[int(np.argmax(scores))]

    def update(
        self, hw: HardwareUnit, job: InvocationJob, completion_time: float, arrival_time: float
    ) -> None:
        latency = completion_time - arrival_time
        reward = -latency
        idx = next(i for i, h in enumerate(self.hw_units) if h.hw_id == hw.hw_id)
        self.counts[idx] += 1
        n = self.counts[idx]
        value = self.values[idx]
        self.values[idx] = value + (reward - value) / n
        self.warm_func[hw.hw_id] = job.func_id


# -----------------------------
# SimPy cluster simulator
# -----------------------------


class ClusterSimulator:
    """
    Discrete-event simulator with:
      - ground-truth service times (affinity-aware)
      - cold-start penalties when function is not warm on the unit
      - queue-aware scheduling hooks for RL / optimizer
    """

    def __init__(
        self,
        jobs: List[InvocationJob],
        hw_units: List[HardwareUnit],
        predictor: GNNPredictor,
        optimizer: Optional[MultiObjectiveOptimizer],
        scheduler,
        use_optimizer: bool = True,
        optimizer_rl_bias: float = 0.5,
        slo_target_ms: float = 100.0,
        enable_cold_start: bool = True,
        cold_start_ms: float = 18.0,
    ) -> None:
        self.env = simpy.Environment()
        self.jobs = sorted(jobs, key=lambda j: j.arrival_time)
        self.hw_units = hw_units
        self.predictor = predictor
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.use_optimizer = use_optimizer and optimizer is not None
        self.optimizer_rl_bias = optimizer_rl_bias
        self.slo_target_ms = slo_target_ms
        self.enable_cold_start = enable_cold_start
        self.cold_start_ms = cold_start_ms
        self.resources: Dict[int, simpy.Resource] = {
            hw.hw_id: simpy.Resource(self.env, capacity=1) for hw in hw_units
        }
        self.job_latencies: List[float] = []
        self.hw_utilization: Dict[int, float] = {hw.hw_id: 0.0 for hw in hw_units}
        self.total_energy: float = 0.0
        self.total_cost: float = 0.0
        self.makespan: float = 0.0
        self.slo_violations: int = 0
        self.decision_times_s: List[float] = []
        self.warm_func: Dict[int, Optional[str]] = {hw.hw_id: None for hw in hw_units}
        self.cold_starts: int = 0
        self.warm_hits: int = 0

    def run(self) -> None:
        self.env.process(self._job_arrival_process())
        self.env.run()
        self.makespan = float(self.env.now)

    def _job_arrival_process(self):
        for job in self.jobs:
            yield self.env.timeout(max(0.0, job.arrival_time - self.env.now))
            self.env.process(self._handle_job(job))

    def _queue_lengths(self) -> Dict[int, int]:
        return {hw_id: len(res.queue) + (1 if res.count >= res.capacity else 0)
                for hw_id, res in self.resources.items()}

    def _is_cold(self, hw: HardwareUnit, job: InvocationJob) -> bool:
        if not self.enable_cold_start:
            return False
        return self.warm_func.get(hw.hw_id) != job.func_id

    def _q_value(self, hw: HardwareUnit) -> float:
        if hasattr(self.scheduler, "values"):
            try:
                idx = next(i for i, h in enumerate(self.scheduler.hw_units) if h.hw_id == hw.hw_id)
                return float(self.scheduler.values[idx])
            except (StopIteration, AttributeError):
                return 0.0
        return 0.0

    def _handle_job(self, job: InvocationJob):
        t0 = self.env.now
        preds = {hw.hw_id: self.predictor.predict(job, hw_type=hw.hw_type) for hw in self.hw_units}
        qlens = self._queue_lengths()

        # Keep RL warm map in sync with simulator
        if hasattr(self.scheduler, "warm_func"):
            self.scheduler.warm_func = dict(self.warm_func)

        try:
            candidate_hw = self.scheduler.select_hw(
                job, now=self.env.now, predictions=preds, queue_lengths=qlens
            )
        except TypeError:
            try:
                candidate_hw = self.scheduler.select_hw(job, now=self.env.now, predictions=preds)
            except TypeError:
                candidate_hw = self.scheduler.select_hw(job, now=self.env.now)

        if self.use_optimizer:
            best_hw = candidate_hw
            best_score = self.optimizer.score_assignment(
                job,
                preds[candidate_hw.hw_id],
                candidate_hw,
                queue_len=float(qlens.get(candidate_hw.hw_id, 0)),
                is_cold=self._is_cold(candidate_hw, job),
            )
            best_score -= self.optimizer_rl_bias * self._q_value(candidate_hw)
            for hw in self.hw_units:
                score = self.optimizer.score_assignment(
                    job,
                    preds[hw.hw_id],
                    hw,
                    queue_len=float(qlens.get(hw.hw_id, 0)),
                    is_cold=self._is_cold(hw, job),
                )
                score -= self.optimizer_rl_bias * self._q_value(hw)
                if score < best_score:
                    best_score = score
                    best_hw = hw
            chosen_hw = best_hw
        else:
            chosen_hw = candidate_hw

        self.decision_times_s.append(max(0.0, self.env.now - t0))
        is_cold = self._is_cold(chosen_hw, job)

        res = self.resources[chosen_hw.hw_id]
        with res.request() as req:
            yield req
            runtime_ms = job.base_duration / effective_speedup(job, chosen_hw)
            if is_cold:
                runtime_ms += self.cold_start_ms
                self.cold_starts += 1
            else:
                self.warm_hits += 1
            runtime_s = max(1e-6, runtime_ms / 1000.0)
            yield self.env.timeout(runtime_s)
            completion_time = self.env.now
            latency = completion_time - job.arrival_time
            self.job_latencies.append(latency)
            if latency * 1000.0 > self.slo_target_ms:
                self.slo_violations += 1

            self.hw_utilization[chosen_hw.hw_id] += runtime_s
            self.total_energy += runtime_ms * chosen_hw.energy_per_ms
            self.total_cost += runtime_ms * chosen_hw.cost_per_ms
            self.warm_func[chosen_hw.hw_id] = job.func_id

            if hasattr(self.scheduler, "update"):
                self.scheduler.update(chosen_hw, job, completion_time, job.arrival_time)


def build_default_cluster(config: Optional[ExperimentConfig] = None) -> List[HardwareUnit]:
    """Backward-compatible small cluster (3 units). Prefer build_experimental_cluster."""
    return [
        HardwareUnit(hw_id=0, hw_type="cpu", speedup=1.0, energy_per_ms=1.0, cost_per_ms=1.0),
        HardwareUnit(hw_id=1, hw_type="cpu", speedup=1.2, energy_per_ms=1.2, cost_per_ms=1.1),
        HardwareUnit(hw_id=2, hw_type="gpu", speedup=3.0, energy_per_ms=2.0, cost_per_ms=2.5),
    ]


def build_cluster_from_config(config: Optional[ExperimentConfig] = None) -> List[HardwareUnit]:
    cfg = config or DEFAULT_CONFIG
    hw_units: List[HardwareUnit] = []
    hw_id = 0
    for _ in range(cfg.num_cpu_units):
        hw_units.append(
            HardwareUnit(
                hw_id=hw_id,
                hw_type="cpu",
                speedup=cfg.cpu_speedup,
                energy_per_ms=cfg.cpu_energy_per_ms,
                cost_per_ms=cfg.cpu_cost_per_ms,
            )
        )
        hw_id += 1
    for _ in range(cfg.num_gpu_units):
        hw_units.append(
            HardwareUnit(
                hw_id=hw_id,
                hw_type="gpu",
                speedup=cfg.gpu_speedup,
                energy_per_ms=cfg.gpu_energy_per_ms,
                cost_per_ms=cfg.gpu_cost_per_ms,
            )
        )
        hw_id += 1
    for _ in range(cfg.num_fpga_units):
        hw_units.append(
            HardwareUnit(
                hw_id=hw_id,
                hw_type="fpga",
                speedup=cfg.fpga_speedup,
                energy_per_ms=cfg.fpga_energy_per_ms,
                cost_per_ms=cfg.fpga_cost_per_ms,
            )
        )
        hw_id += 1
    return hw_units


def run_dandelion_learn_simulation(root: Optional[Path] = None, day: int = 1, max_jobs: int = 2000) -> None:
    cfg = DEFAULT_CONFIG
    trace_dir = get_azure_trace_dir((root / "azure_trace") if root else None)
    jobs = load_azure_invocations(trace_dir, day=day, max_jobs=max_jobs)
    if not jobs:
        raise RuntimeError("No jobs loaded from Azure trace.")

    predictor = GNNPredictor(d_hidden=cfg.gnn_hidden, L=cfg.gnn_layers, device=cfg.device, seed=cfg.seed)
    predictor.fit(
        jobs[: max(1, len(jobs) // 2)],
        epochs=cfg.gnn_epochs,
        lr=cfg.gnn_lr,
        batch_size=cfg.gnn_batch_size,
        weight_decay=cfg.gnn_weight_decay,
        verbose=True,
    )

    hw_units = build_cluster_from_config(cfg)
    optimizer = MultiObjectiveOptimizer(
        alpha=cfg.optimizer_alpha,
        beta=cfg.optimizer_beta,
        gamma=cfg.optimizer_gamma,
        queue_weight=cfg.optimizer_queue_weight,
        cold_weight=cfg.optimizer_cold_weight,
        cold_start_ms=cfg.cold_start_ms,
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

    sim = ClusterSimulator(
        jobs=jobs,
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
    sim.run()

    latencies = np.array(sim.job_latencies)
    if latencies.size == 0:
        print("Simulation produced no completed jobs.")
        return

    p50 = np.percentile(latencies, 50) * 1000
    p95 = np.percentile(latencies, 95) * 1000
    p99 = np.percentile(latencies, 99) * 1000
    throughput = len(latencies) / sim.makespan if sim.makespan > 0 else 0.0
    slo_rate = 1.0 - (sim.slo_violations / len(latencies))

    print(f"Jobs simulated: {len(latencies)}")
    print(f"Latency (ms): p50={p50:.2f}, p95={p95:.2f}, p99={p99:.2f}")
    print(f"Throughput: {throughput:.2f} jobs/s (makespan={sim.makespan:.3f}s)")
    print(f"SLO compliance (@{cfg.slo_target_ms}ms): {100*slo_rate:.1f}%")


if __name__ == "__main__":
    run_dandelion_learn_simulation(Path(".").resolve())
