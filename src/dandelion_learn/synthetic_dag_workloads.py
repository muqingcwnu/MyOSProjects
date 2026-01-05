from __future__ import annotations

"""
Synthetic DAG Workload Generation

Generates synthetic function DAGs for controlled experiments, in addition
to real-world Azure traces.

DAG structure:
- Function workflows as directed acyclic graphs (DAGs)
- Nodes = functions, edges = data dependencies
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from dandelion_learn.dandelion_learn_sim import InvocationJob


@dataclass
class FunctionDAG:
    """
    Represents a function workflow as a DAG.
    
    Structure:
    - G = (V, E) where V = functions, E = data dependencies
    - Each function has input/output data sizes
    - DAG structure determines execution order
    """
    nodes: List[str]  # Function IDs
    edges: List[Tuple[str, str]]  # (source_func, target_func) data dependencies
    data_sizes: Dict[Tuple[str, str], float]  # Edge -> data size (KB)
    function_runtimes: Dict[str, float]  # Function -> base runtime (ms)


def generate_synthetic_dag(
    num_functions: int = 10,
    max_parallelism: int = 3,
    dag_type: str = "chain",
) -> FunctionDAG:
    """
    Generate synthetic function DAGs for controlled experiments.
    
    Args:
        num_functions: Number of functions in the DAG
        max_parallelism: Maximum parallel branches
        dag_type: "chain", "fork_join", "diamond", "random"
    
    Returns:
        FunctionDAG representing the workflow
    """
    nodes = [f"func_{i}" for i in range(num_functions)]
    edges: List[Tuple[str, str]] = []
    data_sizes: Dict[Tuple[str, str], float] = {}
    function_runtimes: Dict[str, float] = {}
    
    if dag_type == "chain":
        # Linear chain: func_0 -> func_1 -> ... -> func_n
        for i in range(num_functions - 1):
            edges.append((nodes[i], nodes[i + 1]))
            data_sizes[(nodes[i], nodes[i + 1])] = np.random.uniform(1.0, 10.0)
    
    elif dag_type == "fork_join":
        # Fork from first function, join at last
        # func_0 -> func_1, func_2, ..., func_{n-2} -> func_{n-1}
        for i in range(1, num_functions - 1):
            edges.append((nodes[0], nodes[i]))
            data_sizes[(nodes[0], nodes[i])] = np.random.uniform(1.0, 10.0)
        for i in range(1, num_functions - 1):
            edges.append((nodes[i], nodes[-1]))
            data_sizes[(nodes[i], nodes[-1])] = np.random.uniform(1.0, 10.0)
    
    elif dag_type == "diamond":
        # Diamond pattern: start -> parallel -> merge -> end
        if num_functions >= 4:
            edges.append((nodes[0], nodes[1]))
            edges.append((nodes[0], nodes[2]))
            edges.append((nodes[1], nodes[3]))
            edges.append((nodes[2], nodes[3]))
            for edge in edges:
                data_sizes[edge] = np.random.uniform(1.0, 10.0)
    
    elif dag_type == "random":
        # Random DAG structure
        for i in range(num_functions - 1):
            # Each function connects to 1-3 random later functions
            num_targets = np.random.randint(1, min(4, num_functions - i))
            targets = np.random.choice(
                nodes[i + 1:], size=num_targets, replace=False
            )
            for target in targets:
                edges.append((nodes[i], target))
                data_sizes[(nodes[i], target)] = np.random.uniform(1.0, 10.0)
    
    # Assign random runtimes to functions
    for node in nodes:
        function_runtimes[node] = np.random.uniform(5.0, 50.0)
    
    return FunctionDAG(nodes=nodes, edges=edges, data_sizes=data_sizes, function_runtimes=function_runtimes)


def dag_to_jobs(
    dag: FunctionDAG,
    num_workflows: int = 100,
    arrival_rate: float = 0.1,
) -> List[InvocationJob]:
    """
    Convert a function DAG into a list of InvocationJob objects.
    
    This simulates multiple workflow executions, respecting DAG dependencies.
    """
    jobs: List[InvocationJob] = []
    job_id = 0
    time_cursor = 0.0
    
    for workflow_id in range(num_workflows):
        # Topological sort for execution order
        executed = set()
        workflow_jobs: List[InvocationJob] = []
        
        # Find root nodes
        incoming = {node: [] for node in dag.nodes}
        for src, dst in dag.edges:
            incoming[dst].append(src)
        
        roots = [node for node in dag.nodes if len(incoming[node]) == 0]
        
        # Execute in topological order
        queue = roots.copy()
        while queue:
            node = queue.pop(0)
            if node in executed:
                continue
            
            # Check if deps are done
            if all(dep in executed for dep in incoming[node]):
                # Create job
                input_size = sum(
                    dag.data_sizes.get((dep, node), 0.0) for dep in incoming[node]
                )
                if input_size == 0:
                    input_size = np.random.uniform(1.0, 5.0)
                
                job = InvocationJob(
                    job_id=job_id,
                    func_id=f"{node}_wf{workflow_id}",
                    arrival_time=time_cursor,
                    input_size=input_size,
                    base_duration=dag.function_runtimes[node],
                )
                workflow_jobs.append(job)
                executed.add(node)
                job_id += 1
                
                # Add children
                for src, dst in dag.edges:
                    if src == node and dst not in executed and dst not in queue:
                        queue.append(dst)
        
        # Add jobs with inter-arrival spacing
        for job in workflow_jobs:
            jobs.append(job)
            time_cursor += np.random.exponential(scale=arrival_rate)
    
    return jobs


def generate_synthetic_workloads() -> Dict[str, List[InvocationJob]]:
    """
    Generate multiple synthetic DAG workloads for evaluation.
    
    Returns:
        Dictionary mapping workload name to list of jobs
    """
    workloads = {}
    
    # Chain DAGs (linear workflows)
    for size in [5, 10, 20]:
        dag = generate_synthetic_dag(num_functions=size, dag_type="chain")
        jobs = dag_to_jobs(dag, num_workflows=50)
        workloads[f"chain_{size}"] = jobs
    
    # Fork-join DAGs (parallel branches)
    for size in [8, 15]:
        dag = generate_synthetic_dag(num_functions=size, dag_type="fork_join")
        jobs = dag_to_jobs(dag, num_workflows=50)
        workloads[f"fork_join_{size}"] = jobs
    
    # Diamond DAGs
    dag = generate_synthetic_dag(num_functions=10, dag_type="diamond")
    jobs = dag_to_jobs(dag, num_workflows=50)
    workloads["diamond"] = jobs
    
    # Random DAGs
    for size in [10, 15]:
        dag = generate_synthetic_dag(num_functions=size, dag_type="random")
        jobs = dag_to_jobs(dag, num_workflows=50)
        workloads[f"random_{size}"] = jobs
    
    return workloads

