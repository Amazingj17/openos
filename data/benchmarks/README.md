# Public benchmark manifests

This directory commits metadata and frozen split manifests, not third-party raw
archives. Raw data is downloaded into the Git-ignored `outputs/` directory and
must match the exact byte size and SHA-256 recorded in the manifest.

## STG rnc50 projection v1

Source dataset:

- Aasish Kumar Sharma and Julian Martin Kunkel, *Standard Task Graph (STG)
  Dataset With JSON Conversions for Workflow Scheduling in Heterogeneous High
  Performance Computing (HPC) Systems*, Zenodo, 2026.
- DOI: <https://doi.org/10.5281/zenodo.18927122>
- License for the JSON conversions, packaging, system configurations and
  documentation: CC BY 4.0.
- Original STG credit: Hiroshi Kasahara and collaborators, Waseda University.

The upstream archive states that the original STG graphs are redistributed with
permission. TriSched does not copy the upstream GitHub solver code because the
repository root has no license, and it does not copy Zenodo record 20419279
benchmark results because that record does not declare rights.

Fetch and independently verify the pinned 125,500-byte archive:

```powershell
python scripts/fetch_stg_benchmark.py
python scripts/fetch_stg_benchmark.py --offline
```

The loader uses an explicit capability-relaxed projection. It preserves DAG
topology, task duration and predecessor output data, but it does not preserve
the upstream CPU/GPU, core-count or memory constraints because the current
TriSched `Scenario` cannot express them. Results must be described as the
"TriSched STG topology projection v1", not as a reproduction of the complete
GrapheonRL heterogeneous system benchmark.

See [`stg-rnc50-hetero-v1.json`](stg-rnc50-hetero-v1.json) for every source and
scenario hash, and [`doc/P1-B01公开基准与许可证.md`](../../doc/P1-B01公开基准与许可证.md)
for the license audit, split protocol and known limitations. The immutable-CI,
ten-instance projection and fault-injection review is recorded in
[`doc/P1-B01独立复核记录.md`](../../doc/P1-B01独立复核记录.md).
