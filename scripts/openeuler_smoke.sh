#!/usr/bin/env bash
set -euo pipefail

REPOSITORY="${1:-/workspace}"
OUTPUT="${2:-/workspace/outputs/p0-10-openeuler}"
VENV="${TRISCHED_VENV:-/opt/trisched-venv}"
GIT_HEAD="${TRISCHED_GIT_HEAD:-unknown}"

mkdir -p "${OUTPUT}"
exec > >(tee "${OUTPUT}/smoke.log") 2>&1

echo "[1/8] recording openEuler container and CPU platform"
{
  cat /etc/os-release
  uname -a
  uname -m
  cat /proc/cpuinfo | sed -n '1,24p'
} | tee "${OUTPUT}/platform.txt"

echo "[2/8] verifying the image-provided Python runtime"
rpm -q python3
python3 --version
python3 -m ensurepip --version

echo "[3/8] creating a clean virtual environment"
python3 -m venv "${VENV}"
PYTHON="${VENV}/bin/python"

cd "${REPOSITORY}"
echo "git_head=${GIT_HEAD}"

echo "[4/8] installing locked dependencies and a non-editable package"
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-120}"
export PIP_RETRIES="${PIP_RETRIES:-5}"
PIP_INSTALL_ARGS=(--no-deps)
if [[ -d "${OUTPUT}/wheelhouse" ]]; then
  PIP_INSTALL_ARGS+=(--find-links "${OUTPUT}/wheelhouse")
fi
NUMPY_WHEEL="${OUTPUT}/wheelhouse/numpy-1.26.4-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
if [[ -f "${NUMPY_WHEEL}" ]]; then
  "${PYTHON}" -m pip install --no-deps "${NUMPY_WHEEL}"
fi
"${PYTHON}" -m pip install "${PIP_INSTALL_ARGS[@]}" -r requirements-lock.txt
"${PYTHON}" -m pip install --no-build-isolation --no-deps .
"${PYTHON}" -m pip check
"${PYTHON}" -m pip freeze > "${OUTPUT}/pip-freeze.txt"

cd /tmp
"${PYTHON}" - <<'PY'
import trisched

module_path = trisched.__file__
assert module_path is not None
assert "site-packages" in module_path
print(f"installed_module={module_path}")
PY

echo "[5/8] running the complete test suite"
cd "${REPOSITORY}"
"${PYTHON}" -m pytest -o addopts="" -q --junitxml="${OUTPUT}/pytest.xml"

echo "[6/8] running the installed pipeline outside the repository"
cd /tmp
"${PYTHON}" -m trisched pipeline \
  --config "${REPOSITORY}/configs/smoke.json" \
  --output "${OUTPUT}/pipeline"

echo "[7/8] loading the checkpoint without retraining"
"${PYTHON}" -m trisched evaluate \
  --config "${REPOSITORY}/configs/smoke.json" \
  --checkpoint "${OUTPUT}/pipeline/masked_mlp.npz" \
  --split test \
  --output "${OUTPUT}/evaluate"

echo "[8/8] validating and hashing structured evidence"
"${PYTHON}" - "${OUTPUT}" "${GIT_HEAD}" <<'PY' | tee "${OUTPUT}/validation.json"
import hashlib
import json
import platform
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

output = Path(sys.argv[1])
git_head = sys.argv[2]
summary = json.loads((output / "pipeline" / "summary.json").read_text())
evaluation = json.loads(
    (output / "evaluate" / "evaluation_summary.json").read_text()
)

expected_policies = ("heft", "cpop", "greedy_eft", "random", "masked_mlp")
for policy in expected_policies:
    assert summary["test"][policy]["count"] == 20
    assert summary["test"][policy]["valid_schedule_rate"] == 1.0
assert summary["test"]["masked_mlp"]["mean_ratio"] == 1.0
assert evaluation["dataset"]["count"] == 20
assert evaluation["metrics"]["masked_mlp"]["mean_ratio"] == 1.0
assert evaluation["metrics"]["masked_mlp"]["valid_schedule_rate"] == 1.0

manifests = summary["datasets"]
hashes = {
    split: set(manifests[split]["scenario_hashes"])
    for split in ("train", "validation", "test")
}
assert not hashes["train"] & hashes["validation"]
assert not hashes["train"] & hashes["test"]
assert not hashes["validation"] & hashes["test"]

checkpoint = output / "pipeline" / "masked_mlp.npz"
checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
assert checkpoint_sha256 == evaluation["checkpoint"]["sha256"]
junit_root = ET.parse(output / "pytest.xml").getroot()
if "tests" in junit_root.attrib:
    test_count = int(junit_root.attrib["tests"])
else:
    test_count = sum(
        int(test_suite.attrib.get("tests", 0))
        for test_suite in junit_root.findall("testsuite")
    )
assert test_count > 0

evidence = {
    "status": "pass",
    "git_head": git_head,
    "python": platform.python_version(),
    "platform": platform.platform(),
    "machine": platform.machine(),
    "test_count": test_count,
    "dataset_counts": {split: manifests[split]["count"] for split in manifests},
    "dataset_hash_intersections": {
        "train_validation": 0,
        "train_test": 0,
        "validation_test": 0,
    },
    "masked_mlp_mean_ratio": evaluation["metrics"]["masked_mlp"][
        "mean_ratio"
    ],
    "checkpoint_sha256": checkpoint_sha256,
}
print(json.dumps(evidence, indent=2, sort_keys=True))
PY

sha256sum \
  "${REPOSITORY}/requirements-lock.txt" \
  "${REPOSITORY}/LICENSE" \
  "${OUTPUT}/pipeline/masked_mlp.npz" \
  > "${OUTPUT}/sha256.txt"

echo "openEuler CPU smoke: PASS"
