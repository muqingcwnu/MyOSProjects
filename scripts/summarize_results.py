"""Summarize evaluation CSVs under results_fixed/."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results_fixed"


def main() -> None:
    base = RESULTS_DIR
    if not (base / "performance_results.csv").exists():
        raise FileNotFoundError(f"performance_results.csv not found in {base}")

    print(f"Results dir: {base}")
    for f in sorted(base.glob("*.csv")):
        print(f"  - {f.name} ({f.stat().st_size / 1024:.1f} KB)")

    perf = pd.read_csv(base / "performance_results.csv")
    print(f"\nPerformance: {len(perf)} rows")
    print("Schedulers:", ", ".join(sorted(perf["scheduler"].unique())))
    print("Workloads:", ", ".join(sorted(perf["workload"].unique())))

    agg = {
        "p50": ("p50_latency_ms", "mean"),
        "p95": ("p95_latency_ms", "mean"),
        "p99": ("p99_latency_ms", "mean"),
        "mean": ("mean_latency_ms", "mean"),
        "thr": ("throughput", "mean"),
        "energy": ("energy_per_job", "mean"),
        "cost": ("cost_per_job", "mean"),
    }
    if "slo_compliance_rate" in perf.columns:
        agg["slo"] = ("slo_compliance_rate", "mean")
    g = perf.groupby("scheduler").agg(**{k: v for k, v in agg.items()}).sort_values("p99")
    print("\n=== Mean across all workloads/runs ===")
    print(g.round(3).to_string())

    if "FIFO" in g.index and "Dandelion-Learn" in g.index:
        fifo, dl = g.loc["FIFO"], g.loc["Dandelion-Learn"]
        print("\n=== Dandelion-Learn vs FIFO ===")
        print(
            f"p99: {fifo.p99:.3f} -> {dl.p99:.3f} ms  "
            f"({(1 - dl.p99 / fifo.p99) * 100:.1f}% lower)"
        )
        print(f"throughput: {fifo.thr:.2f} -> {dl.thr:.2f}")
        print(
            f"energy/job: {fifo.energy:.3f} -> {dl.energy:.3f}  "
            f"({(1 - dl.energy / fifo.energy) * 100:.1f}% lower)"
        )
        print(
            f"cost/job: {fifo.cost:.3f} -> {dl.cost:.3f}  "
            f"({(1 - dl.cost / fifo.cost) * 100:.1f}% lower)"
        )
        if "slo" in g.columns:
            print(f"SLO compliance: {dl.slo * 100:.1f}%")

    abl = pd.read_csv(base / "ablation_studies.csv")
    print("\n=== Ablation ===")
    cols = [
        c
        for c in [
            "scheduler",
            "components",
            "p99_latency_ms",
            "throughput",
            "energy_per_job",
            "slo_compliance_rate",
        ]
        if c in abl.columns
    ]
    print(abl[cols].round(3).to_string(index=False))

    scal = pd.read_csv(base / "scalability_overhead.csv")
    print("\n=== Scalability ===")
    scols = [
        c
        for c in [
            "num_jobs",
            "p99_latency_ms",
            "throughput",
            "overhead_percentage",
            "gnn_prediction_time_ms",
        ]
        if c in scal.columns
    ]
    print(scal[scols].round(3).to_string(index=False))

    sec = pd.read_csv(base / "security_analysis.csv")
    print("\n=== Security (head) ===")
    print(sec.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
