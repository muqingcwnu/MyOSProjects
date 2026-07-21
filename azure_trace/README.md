# Azure Functions 2019 traces

Updated: **2026-07-21 07:51 UTC**

This folder holds the Azure Public Dataset (Functions 2019) CSVs used by Dandelion-Learn.

Files are stored with **Git LFS**. On GitHub you will see small pointer files (~130 bytes);
the real CSVs (~2.14 GB total) are fetched with:

```bash
git lfs install
git lfs pull
```

## Contents

| Pattern | Files | Role |
|---------|------:|------|
| `invocations_per_function_md.anon.d*.csv` | 14 | Per-function invocations |
| `function_durations_percentiles.anon.d*.csv` | 14 | Duration percentiles |
| `app_memory_percentiles.anon.d*.csv` | 12 | Memory percentiles |

See `MANIFEST.csv` for exact sizes and SHA-256 hashes of the local dataset objects.
