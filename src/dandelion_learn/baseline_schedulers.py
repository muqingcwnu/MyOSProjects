from __future__ import annotations

"""
Baseline heuristic schedulers for comparison with Dandelion-Learn.

These implement common scheduling policies used in production serverless
platforms and research systems.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np

from dandelion_learn.dandelion_learn_sim import HardwareUnit, InvocationJob, predicted_speedup


def _expected_latency_ms(service_ms: float, queue_len: float) -> float:
    """Sojourn-time proxy: service plus wait proportional to queue depth."""
    return float(service_ms) * (1.0 + max(0.0, float(queue_len)))


class BaseScheduler(ABC):
    """
    Abstract base class for all schedulers.
    """

    def __init__(self, hw_units: List[HardwareUnit]) -> None:
        self.hw_units = hw_units

    @abstractmethod
    def select_hw(
        self,
        job: InvocationJob,
        now: float,
        predictions=None,
        queue_lengths: Optional[Dict[int, int]] = None,
    ) -> HardwareUnit:
        """Select a hardware unit for the given job."""
        pass

    def update(
        self, hw: HardwareUnit, job: InvocationJob, completion_time: float, arrival_time: float
    ) -> None:
        """
        Optional callback after job completion for learning schedulers.
        Baseline schedulers can ignore this.
        """
        pass


class FIFOScheduler(BaseScheduler):
    """
    First-In-First-Out scheduler: assigns jobs to hardware units in
    round-robin order, ignoring job characteristics.
    """

    def __init__(self, hw_units: List[HardwareUnit]) -> None:
        super().__init__(hw_units)
        self.next_hw_idx = 0

    def select_hw(
        self,
        job: InvocationJob,
        now: float,
        predictions=None,
        queue_lengths: Optional[Dict[int, int]] = None,
    ) -> HardwareUnit:
        hw = self.hw_units[self.next_hw_idx]
        self.next_hw_idx = (self.next_hw_idx + 1) % len(self.hw_units)
        return hw


class RandomScheduler(BaseScheduler):
    """Random scheduler: assigns each job to a uniformly random hardware unit."""

    def select_hw(
        self,
        job: InvocationJob,
        now: float,
        predictions=None,
        queue_lengths: Optional[Dict[int, int]] = None,
    ) -> HardwareUnit:
        idx = np.random.randint(len(self.hw_units))
        return self.hw_units[idx]


class RoundRobinScheduler(BaseScheduler):
    """Round-robin scheduler: cycles through hardware units in order."""

    def __init__(self, hw_units: List[HardwareUnit]) -> None:
        super().__init__(hw_units)
        self.counter = 0

    def select_hw(
        self,
        job: InvocationJob,
        now: float,
        predictions=None,
        queue_lengths: Optional[Dict[int, int]] = None,
    ) -> HardwareUnit:
        idx = self.counter % len(self.hw_units)
        self.counter += 1
        return self.hw_units[idx]


class ShortestJobFirstScheduler(BaseScheduler):
    """
    Shortest expected sojourn time: GNN-predicted service time + queue wait.

    Without queue awareness, affinity-aware SJF herds onto one accelerator and
    collapses under bursty arrivals (unusable baseline for fair comparison).
    """

    def __init__(self, hw_units: List[HardwareUnit], predictor) -> None:
        super().__init__(hw_units)
        self.predictor = predictor

    def select_hw(
        self,
        job: InvocationJob,
        now: float,
        predictions=None,
        queue_lengths: Optional[Dict[int, int]] = None,
    ) -> HardwareUnit:
        qlens = queue_lengths or {}
        best_hw = self.hw_units[0]
        best_lat = float("inf")
        for hw in self.hw_units:
            pred = self.predictor.predict(job, hw_type=hw.hw_type)
            service = pred.runtime_ms / predicted_speedup(pred, hw)
            lat = _expected_latency_ms(service, qlens.get(hw.hw_id, 0))
            if lat < best_lat:
                best_lat = lat
                best_hw = hw
        return best_hw


class LocalityAwareScheduler(BaseScheduler):
    """
    Locality-aware scheduler: prefers warm units, but sheds load when the
    preferred unit's queue is much deeper than the least-loaded unit.
    """

    def __init__(self, hw_units: List[HardwareUnit]) -> None:
        super().__init__(hw_units)
        self.func_to_hw: dict[str, int] = {}

    def select_hw(
        self,
        job: InvocationJob,
        now: float,
        predictions=None,
        queue_lengths: Optional[Dict[int, int]] = None,
    ) -> HardwareUnit:
        qlens = queue_lengths or {hw.hw_id: 0 for hw in self.hw_units}
        least = min(self.hw_units, key=lambda h: qlens.get(h.hw_id, 0))

        if job.func_id in self.func_to_hw:
            preferred_id = self.func_to_hw[job.func_id]
            for hw in self.hw_units:
                if hw.hw_id == preferred_id:
                    # Shed if preferred is heavily congested vs least-loaded
                    if qlens.get(hw.hw_id, 0) <= qlens.get(least.hw_id, 0) + 2:
                        return hw
                    return least

        return least

    def update(
        self, hw: HardwareUnit, job: InvocationJob, completion_time: float, arrival_time: float
    ) -> None:
        self.func_to_hw[job.func_id] = hw.hw_id


class SinanScheduler(BaseScheduler):
    """
    Sinan-style scheduler: prefers hardware with lower current load / queue.
    """

    def __init__(self, hw_units: List[HardwareUnit]) -> None:
        super().__init__(hw_units)
        self.hw_load: dict[int, float] = {hw.hw_id: 0.0 for hw in hw_units}
        self.hw_queue_length: dict[int, int] = {hw.hw_id: 0 for hw in hw_units}

    def select_hw(
        self,
        job: InvocationJob,
        now: float,
        predictions=None,
        queue_lengths: Optional[Dict[int, int]] = None,
    ) -> HardwareUnit:
        # Prefer live simulator queues when available
        if queue_lengths is not None:
            best_hw = min(
                self.hw_units,
                key=lambda hw: self.hw_load[hw.hw_id] + float(queue_lengths.get(hw.hw_id, 0)),
            )
        else:
            best_hw = min(
                self.hw_units,
                key=lambda hw: self.hw_load[hw.hw_id] + self.hw_queue_length[hw.hw_id],
            )
        self.hw_queue_length[best_hw.hw_id] += 1
        return best_hw

    def update(
        self, hw: HardwareUnit, job: InvocationJob, completion_time: float, arrival_time: float
    ) -> None:
        runtime = completion_time - arrival_time
        self.hw_load[hw.hw_id] = 0.9 * self.hw_load[hw.hw_id] + 0.1 * runtime
        self.hw_queue_length[hw.hw_id] = max(0, self.hw_queue_length[hw.hw_id] - 1)


class FiferScheduler(BaseScheduler):
    """
    Fifer-style: cost-aware score over expected sojourn time (queue + service).
    """

    def __init__(self, hw_units: List[HardwareUnit], predictor) -> None:
        super().__init__(hw_units)
        self.predictor = predictor
        self.cost_weight = 0.3

    def select_hw(
        self,
        job: InvocationJob,
        now: float,
        predictions=None,
        queue_lengths: Optional[Dict[int, int]] = None,
    ) -> HardwareUnit:
        qlens = queue_lengths or {}
        best_hw = self.hw_units[0]
        best_score = float("inf")

        for hw in self.hw_units:
            pred = self.predictor.predict(job, hw_type=hw.hw_type)
            service = pred.runtime_ms / predicted_speedup(pred, hw)
            latency = _expected_latency_ms(service, qlens.get(hw.hw_id, 0))
            cost = service * hw.cost_per_ms
            score = latency + self.cost_weight * cost
            if score < best_score:
                best_score = score
                best_hw = hw

        return best_hw


class XFaaSScheduler(BaseScheduler):
    """
    X-FaaS-style: co-locate when the preferred unit is not congested; else
    pick minimum expected sojourn time.
    """

    def __init__(self, hw_units: List[HardwareUnit], predictor) -> None:
        super().__init__(hw_units)
        self.predictor = predictor
        self.func_locations: dict[str, int] = {}

    def select_hw(
        self,
        job: InvocationJob,
        now: float,
        predictions=None,
        queue_lengths: Optional[Dict[int, int]] = None,
    ) -> HardwareUnit:
        qlens = queue_lengths or {}

        def expected(hw: HardwareUnit) -> float:
            pred = self.predictor.predict(job, hw_type=hw.hw_type)
            service = pred.runtime_ms / predicted_speedup(pred, hw)
            return _expected_latency_ms(service, qlens.get(hw.hw_id, 0))

        if job.func_id in self.func_locations:
            preferred_id = self.func_locations[job.func_id]
            for hw in self.hw_units:
                if hw.hw_id == preferred_id:
                    least_q = min(qlens.get(h.hw_id, 0) for h in self.hw_units)
                    if qlens.get(hw.hw_id, 0) <= least_q + 2:
                        return hw
                    break

        return min(self.hw_units, key=expected)

    def update(
        self, hw: HardwareUnit, job: InvocationJob, completion_time: float, arrival_time: float
    ) -> None:
        self.func_locations[job.func_id] = hw.hw_id


class FIRMScheduler(BaseScheduler):
    """
    FIRM-style: fairness + expected sojourn time (queue-aware).
    """

    def __init__(self, hw_units: List[HardwareUnit], predictor) -> None:
        super().__init__(hw_units)
        self.predictor = predictor
        self.hw_job_count: dict[int, int] = {hw.hw_id: 0 for hw in hw_units}
        self.fairness_weight = 0.4

    def select_hw(
        self,
        job: InvocationJob,
        now: float,
        predictions=None,
        queue_lengths: Optional[Dict[int, int]] = None,
    ) -> HardwareUnit:
        qlens = queue_lengths or {}
        best_hw = self.hw_units[0]
        best_score = float("inf")

        max_jobs = max(self.hw_job_count.values()) if self.hw_job_count.values() else 1
        fairness_scores = {
            hw.hw_id: 1.0 - (self.hw_job_count[hw.hw_id] / max(max_jobs, 1))
            for hw in self.hw_units
        }

        # Normalize latency so fairness weight is meaningful
        lats = []
        for hw in self.hw_units:
            pred = self.predictor.predict(job, hw_type=hw.hw_type)
            service = pred.runtime_ms / predicted_speedup(pred, hw)
            lats.append(_expected_latency_ms(service, qlens.get(hw.hw_id, 0)))
        max_lat = max(lats) if lats else 1.0

        for hw, lat in zip(self.hw_units, lats):
            lat_n = lat / (max_lat + 1e-9)
            score = (1 - self.fairness_weight) * lat_n + self.fairness_weight * (
                1 - fairness_scores[hw.hw_id]
            )
            if score < best_score:
                best_score = score
                best_hw = hw

        self.hw_job_count[best_hw.hw_id] += 1
        return best_hw
