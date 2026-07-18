param(
    [ValidateSet("A", "B")]
    [string]$Presenter = "A",
    [string]$OutputRoot = ""
)

$ErrorActionPreference = "Stop"
$repository = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$gitHead = (git -C $repository rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $gitHead.Length -ne 40) {
    throw "Cannot resolve the repository commit"
}
$gitStatus = @(git -C $repository status --porcelain --untracked-files=all)
if ($LASTEXITCODE -ne 0) {
    throw "Cannot inspect the repository working tree"
}
if ($gitStatus.Count -ne 0) {
    throw "Accepted demo evidence requires a clean Git working tree"
}

if (-not $OutputRoot) {
    $timestamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
    $OutputRoot = Join-Path $repository "outputs\p0-13-demo-$($Presenter.ToLower())-$timestamp"
}
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$log = Join-Path $OutputRoot "demo.log"

function Invoke-PythonChecked {
    param(
        [string]$Label,
        [string[]]$Arguments
    )
    "[$Label] python $($Arguments -join ' ')" | Tee-Object -FilePath $log -Append
    $lines = & python @Arguments 2>&1
    $exitCode = $LASTEXITCODE
    $lines | Tee-Object -FilePath $log -Append
    if ($exitCode -ne 0) {
        throw "$Label failed with exit code $exitCode"
    }
}

Push-Location $repository
try {
Invoke-PythonChecked -Label "full regression" -Arguments @("-m", "pytest")

$generated = Join-Path $OutputRoot "generated"
Invoke-PythonChecked -Label "scenario generation" -Arguments @(
    "-m", "trisched", "generate",
    "--config", "configs/smoke.json",
    "--output", $generated
)

$validScenario = Join-Path $generated "test\test-0000.json"
Invoke-PythonChecked -Label "valid scenario contract" -Arguments @(
    "-m", "trisched", "validate-scenario",
    "--input", $validScenario
)

"[expected failure] structured invalid-input diagnostic" |
    Tee-Object -FilePath $log -Append
$invalidLines = & python -m trisched validate-scenario `
    --input tests/fixtures/invalid/scenario_cases.json 2>&1
$invalidExit = $LASTEXITCODE
$invalidLines | Tee-Object -FilePath $log -Append
if ($invalidExit -ne 2 -or ($invalidLines -join "`n") -notmatch '"code": "missing_field"') {
    throw "Invalid scenario did not produce the frozen exit-2 diagnostic"
}

$pipeline = Join-Path $OutputRoot "pipeline"
Invoke-PythonChecked -Label "synthetic smoke pipeline" -Arguments @(
    "-m", "trisched", "pipeline",
    "--config", "configs/smoke.json",
    "--output", $pipeline
)

$evaluation = Join-Path $OutputRoot "checkpoint-validation"
Invoke-PythonChecked -Label "checkpoint-only recovery" -Arguments @(
    "-m", "trisched", "evaluate",
    "--config", "configs/smoke.json",
    "--checkpoint", (Join-Path $pipeline "masked_mlp.npz"),
    "--split", "validation",
    "--output", $evaluation
)

$summaryPath = Join-Path $pipeline "summary.json"
$evaluationPath = Join-Path $evaluation "evaluation_summary.json"
$summary = Get-Content $summaryPath -Raw | ConvertFrom-Json
$evaluationSummary = Get-Content $evaluationPath -Raw | ConvertFrom-Json
$result = [ordered]@{
    format_version = 1
    task = "P0-13-DEMO"
    presenter = $Presenter
    source_commit = $gitHead
    started_from_clean_worktree = $true
    public_stg_test_accessed = $false
    synthetic_smoke_test_only = $true
    expected_failure_exit_code = $invalidExit
    expected_failure_code = "missing_field"
    pipeline_test_mean_ratio = $summary.primary_metric.value
    pipeline_test_failure_count = $summary.test.masked_mlp.failure_count
    checkpoint_validation_mean_ratio = $evaluationSummary.metrics.masked_mlp.mean_ratio
    checkpoint_validation_failure_count = $evaluationSummary.metrics.masked_mlp.failure_count
    checkpoint_sha256 = (Get-FileHash `
        (Join-Path $pipeline "masked_mlp.npz") -Algorithm SHA256).Hash.ToLower()
    log = "demo.log"
}
$resultPath = Join-Path $OutputRoot "demo_result.json"
$result | ConvertTo-Json -Depth 5 | Set-Content -Encoding utf8 $resultPath

Write-Host "Demo passed for presenter $Presenter"
Write-Host "Evidence: $resultPath"
} finally {
    Pop-Location
}
