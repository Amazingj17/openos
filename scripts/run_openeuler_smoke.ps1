param(
    [string]$Image = "openeuler/openeuler:24.03-lts-sp4@sha256:17c15554be2a5bc46023acb6e04d609d77642b8c20e236e88deb18e41ae4558e"
)

$ErrorActionPreference = "Stop"
$repository = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$output = Join-Path $repository "outputs\p0-10-openeuler"
$gitHead = (git -C $repository rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $gitHead.Length -ne 40) {
    throw "Cannot resolve the repository commit"
}

docker info | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Docker Linux engine is not available. Start Docker Desktop first."
}

docker image inspect $Image | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Pinned openEuler image is missing. Run: docker pull $Image"
}

New-Item -ItemType Directory -Force -Path $output | Out-Null
docker image inspect $Image --format '{{json .RepoDigests}}' |
    Set-Content -Encoding utf8 (Join-Path $output "image-digest.json")

$numpyWheel = Join-Path $output "wheelhouse\numpy-1.26.4-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
if (Test-Path -LiteralPath $numpyWheel) {
    $expected = "666dbfb6ec68962c033a450943ded891bed2d54e6755e35e5835d63f4f6931d5"
    $actual = (Get-FileHash $numpyWheel -Algorithm SHA256).Hash.ToLower()
    if ($actual -ne $expected) {
        throw "Local NumPy wheel is incomplete or corrupt: $numpyWheel"
    }
}

docker run --rm `
    --name trisched-openeuler-smoke `
    --env "TRISCHED_GIT_HEAD=$gitHead" `
    --mount "type=bind,source=$repository,target=/workspace" `
    $Image `
    bash /workspace/scripts/openeuler_smoke.sh `
    /workspace /workspace/outputs/p0-10-openeuler

if ($LASTEXITCODE -ne 0) {
    throw "openEuler smoke failed with exit code $LASTEXITCODE"
}

Write-Host "openEuler evidence: $output"
