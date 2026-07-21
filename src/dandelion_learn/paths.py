"""
Project path helpers.

Azure Functions 2019 traces must live directly under:

    <project_root>/azure_trace/invocations_per_function_md.anon.dXX.csv
    <project_root>/azure_trace/function_durations_percentiles.anon.dXX.csv
    <project_root>/azure_trace/app_memory_percentiles.anon.dXX.csv

Do NOT point at AzurePublicDataset-master/ (docs only).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def project_root() -> Path:
    """Return repo root (parent of src/)."""
    # .../src/dandelion_learn/paths.py -> parents[2] = project root
    return Path(__file__).resolve().parents[2]


def get_azure_trace_dir(explicit: Optional[Path] = None) -> Path:
    """
    Resolve the Azure 2019 trace directory containing extracted CSVs.

    Search order:
      1) explicit argument
      2) <cwd>/azure_trace
      3) <project_root>/azure_trace
    """
    candidates = []
    if explicit is not None:
        candidates.append(Path(explicit))
    candidates.append(Path.cwd() / "azure_trace")
    candidates.append(project_root() / "azure_trace")

    for cand in candidates:
        if _looks_like_trace_dir(cand):
            return cand.resolve()

    tried = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "Azure 2019 function traces not found.\n"
        f"Tried: {tried}\n"
        "Expected files like:\n"
        "  azure_trace/invocations_per_function_md.anon.d01.csv\n"
        "Extract azurefunctions_dataset2019_*.tar.xz into azure_trace/."
    )


def _looks_like_trace_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    sample = path / "invocations_per_function_md.anon.d01.csv"
    if not sample.exists():
        return False
    # Reject Git LFS stubs
    try:
        head = sample.read_text(encoding="utf-8", errors="ignore")[:200]
        if "git-lfs.github.com" in head:
            return False
    except OSError:
        return False
    # Real day-01 file is ~100MB+
    return sample.stat().st_size > 1_000_000


def require_azure_day(trace_dir: Path, day: int) -> Path:
    """Return path to invocations CSV for a day, or raise a clear error."""
    path = Path(trace_dir) / f"invocations_per_function_md.anon.d{day:02d}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing Azure invocations file: {path}\n"
            f"Place extracted 2019 CSVs in: {trace_dir}"
        )
    return path
