from __future__ import annotations

"""
Baseline heuristic schedulers for comparison with Dandelion-Learn.

These implement common scheduling policies used in production serverless
platforms and research systems.
"""

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from dandelion_learn.dandelion_learn_sim import HardwareUnit, InvocationJob


class BaseScheduler(ABC):
    """
    Abstract base class for all schedulers.
    """

    def __init__(self, hw_units: List[HardwareUnit]) -> None:
        self.hw_units = hw_units

    @abstractmethod
    def select_hw(self, job: InvocationJob, now: float) -> HardwareUnit:
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

    This mimics simple queue-based policies used in many production systems.
    """

    def __init__(self, hw_units: List[HardwareUnit]) -> None:
        super().__init__(hw_units)
        self.next_hw_idx = 0

    def select_hw(self, job: InvocationJob, now: float) -> HardwareUnit:
        hw = self.hw_units[self.next_hw_idx]
        self.next_hw_idx = (self.next_hw_idx + 1) % len(self.hw_units)
        return hw


class RandomScheduler(BaseScheduler):
    """
    Random scheduler: assigns each job to a uniformly random hardware unit.

    This represents a baseline with no intelligence.
    """

    def select_hw(self, job: InvocationJob, now: float) -> HardwareUnit:
        idx = np.random.randint(len(self.hw_units))
        return self.hw_units[idx]


class RoundRobinScheduler(BaseScheduler):
    """
    Round-robin scheduler: cycles through hardware units in order.

    Similar to FIFO but explicitly cycles through all units before repeating.
    """

    def __init__(self, hw_units: List[HardwareUnit]) -> None:
        super().__init__(hw_units)
        self.counter = 0

    def select_hw(self, job: InvocationJob, now: float) -> HardwareUnit:
        idx = self.counter % len(self.hw_units)
        self.counter += 1
        return self.hw_units[idx]


class ShortestJobFirstScheduler(BaseScheduler):
    """
    Shortest Job First (SJF): assigns jobs to the hardware unit that
    minimizes predicted runtime.

    This requires runtime predictions, so it's a "clairvoyant" baseline
    that uses perfect knowledge of job durations.
    """

    def __init__(self, hw_units: List[HardwareUnit], predictor) -> None:
        super().__init__(hw_units)
        self.predictor = predictor

    def select_hw(self, job: InvocationJob, now: float) -> HardwareUnit:
        best_hw = self.hw_units[0]
        best_runtime = float("inf")
        for hw in self.hw_units:
            pred = self.predictor.predict(job, hw_type=hw.hw_type)
            runtime = pred.runtime_ms / hw.speedup
            if runtime < best_runtime:
                best_runtime = runtime
                best_hw = hw
        return best_hw


class LocalityAwareScheduler(BaseScheduler):
    """
    Locality-aware scheduler: prefers hardware units that are already
    "warm" (have recently executed similar functions).

    This mimics heuristic schedulers that try to exploit data locality
    and reduce cold starts.
    """

    def __init__(self, hw_units: List[HardwareUnit]) -> None:
        super().__init__(hw_units)
        self.func_to_hw: dict[str, int] = {}  # func_id -> preferred hw_id

    def select_hw(self, job: InvocationJob, now: float) -> HardwareUnit:
        # Prefer same hardware if we've seen this function
        if job.func_id in self.func_to_hw:
            preferred_id = self.func_to_hw[job.func_id]
            for hw in self.hw_units:
                if hw.hw_id == preferred_id:
                    return hw
        # Otherwise round-robin
        idx = len(self.func_to_hw) % len(self.hw_units)
        return self.hw_units[idx]

    def update(
        self, hw: HardwareUnit, job: InvocationJob, completion_time: float, arrival_time: float
    ) -> None:
        # Remember where we ran this function
        self.func_to_hw[job.func_id] = hw.hw_id


class SinanScheduler(BaseScheduler):
    """
    Sinan-style scheduler: Uses workload-aware scheduling with burst detection.
    
    Based on Sinan: ML-based scheduling for serverless computing.
    Uses simple heuristics: prefers hardware with lower current load.
    """
    
    def __init__(self, hw_units: List[HardwareUnit]) -> None:
        super().__init__(hw_units)
        self.hw_load: dict[int, float] = {hw.hw_id: 0.0 for hw in hw_units}
        self.hw_queue_length: dict[int, int] = {hw.hw_id: 0 for hw in hw_units}
    
    def select_hw(self, job: InvocationJob, now: float) -> HardwareUnit:
        # Pick hardware with lowest load
        best_hw = min(
            self.hw_units,
            key=lambda hw: self.hw_load[hw.hw_id] + self.hw_queue_length[hw.hw_id]
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
    Fifer-style scheduler: Function-aware scheduling with cost optimization.
    
    Based on Fifer: Cost-aware scheduling for serverless functions.
    Balances latency and cost by preferring cheaper hardware when latency is acceptable.
    """
    
    def __init__(self, hw_units: List[HardwareUnit], predictor) -> None:
        super().__init__(hw_units)
        self.predictor = predictor
        self.cost_weight = 0.3  # Weight for cost vs latency
    
    def select_hw(self, job: InvocationJob, now: float) -> HardwareUnit:
        best_hw = self.hw_units[0]
        best_score = float("inf")
        
        for hw in self.hw_units:
            pred = self.predictor.predict(job, hw_type=hw.hw_type)
            latency = pred.runtime_ms / hw.speedup
            cost = pred.runtime_ms * hw.cost_per_ms
            
            # Score = latency + cost_weight * cost
            score = latency + self.cost_weight * cost
            if score < best_score:
                best_score = score
                best_hw = hw
        
        return best_hw


class XFaaSScheduler(BaseScheduler):
    """
    X-FaaS-style scheduler: Cross-function optimization with DAG awareness.
    
    Based on X-FaaS: Cross-function optimization in serverless computing.
    Considers function dependencies and prefers co-location.
    """
    
    def __init__(self, hw_units: List[HardwareUnit], predictor) -> None:
        super().__init__(hw_units)
        self.predictor = predictor
        self.func_locations: dict[str, int] = {}  # func_id -> hw_id
    
    def select_hw(self, job: InvocationJob, now: float) -> HardwareUnit:
        # Prefer same hardware for co-location
        if job.func_id in self.func_locations:
            preferred_id = self.func_locations[job.func_id]
            for hw in self.hw_units:
                if hw.hw_id == preferred_id:
                    return hw
        
        # Otherwise use shortest job first
        best_hw = self.hw_units[0]
        best_runtime = float("inf")
        for hw in self.hw_units:
            pred = self.predictor.predict(job, hw_type=hw.hw_type)
            runtime = pred.runtime_ms / hw.speedup
            if runtime < best_runtime:
                best_runtime = runtime
                best_hw = hw
        
        return best_hw
    
    def update(
        self, hw: HardwareUnit, job: InvocationJob, completion_time: float, arrival_time: float
    ) -> None:
        # Remember function location
        self.func_locations[job.func_id] = hw.hw_id


class FIRMScheduler(BaseScheduler):
    """
    FIRM-style scheduler: Hybrid heuristic with fairness and resource management.
    
    Based on FIRM: Fair and Intelligent Resource Management for serverless computing.
    Balances fairness (equal load distribution) with performance (shortest job first).
    """
    
    def __init__(self, hw_units: List[HardwareUnit], predictor) -> None:
        super().__init__(hw_units)
        self.predictor = predictor
        self.hw_job_count: dict[int, int] = {hw.hw_id: 0 for hw in hw_units}
        self.fairness_weight = 0.4  # Weight for fairness vs performance
    
    def select_hw(self, job: InvocationJob, now: float) -> HardwareUnit:
        best_hw = self.hw_units[0]
        best_score = float("inf")
        
        # Fairness = inverse of job count
        max_jobs = max(self.hw_job_count.values()) if self.hw_job_count.values() else 1
        fairness_scores = {
            hw.hw_id: 1.0 - (self.hw_job_count[hw.hw_id] / max(max_jobs, 1))
            for hw in self.hw_units
        }
        
        for hw in self.hw_units:
            pred = self.predictor.predict(job, hw_type=hw.hw_type)
            latency = pred.runtime_ms / hw.speedup
            
            # Score = weighted combo of latency and fairness
            score = (1 - self.fairness_weight) * latency + self.fairness_weight * (1 - fairness_scores[hw.hw_id])
            
            if score < best_score:
                best_score = score
                best_hw = hw
        
        self.hw_job_count[best_hw.hw_id] += 1
        return best_hw

