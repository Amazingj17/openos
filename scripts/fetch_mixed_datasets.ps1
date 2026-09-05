$ErrorActionPreference = "Stop"

$repository = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repository

$python = Join-Path $repository ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    $python = "python"
}

Write-Host "[1/3] Fetching and verifying STG Zenodo archive"
& $python scripts/fetch_stg_benchmark.py
if ($LASTEXITCODE -ne 0) { throw "STG download or verification failed" }

$dagbench = Join-Path $repository "outputs\datasets\dagbench"
$dagbenchCommit = "e69984fcf48f3c66bd9571a9d50591b61de42722"
Write-Host "[2/3] Fetching pinned DAGBench"
if (-not (Test-Path -LiteralPath (Join-Path $dagbench ".git"))) {
    git clone --no-checkout https://github.com/ANRGUSC/dagbench.git $dagbench
    if ($LASTEXITCODE -ne 0) { throw "DAGBench clone failed" }
}
git -C $dagbench fetch --depth 1 origin $dagbenchCommit
if ($LASTEXITCODE -ne 0) { throw "DAGBench pinned commit fetch failed" }
git -C $dagbench checkout --detach $dagbenchCommit
if ($LASTEXITCODE -ne 0) { throw "DAGBench pinned checkout failed" }
$actualCommit = (git -C $dagbench rev-parse HEAD).Trim()
if ($actualCommit -ne $dagbenchCommit) { throw "DAGBench commit mismatch" }

Write-Host "[3/3] Fetching portable Topology Zoo graph skeletons"
& $python scripts/fetch_topology_zoo_graphs.py
if ($LASTEXITCODE -ne 0) { throw "Topology Zoo download failed" }

Write-Host "All mixed-dataset sources are ready."
