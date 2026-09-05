from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path


COMMIT = "ae88b808d71e0b1852186aad2b0ad694b6b5c864"
TOPOLOGIES = (
    "Aconet", "Arn", "Arpanet19728", "Bics", "Bren", "BtLatinAmerica",
    "Canerie", "Carnet", "Cesnet200511", "Cesnet200603", "Claranet", "Cwix",
    "Cynet", "EliBackbone", "Evolink", "Gambia", "Garr199905", "GtsPoland",
    "GtsSlovakia", "Janetbackbone", "LambdaNet", "Litnet", "Marnet",
    "NetworkUsa", "Niif", "Oxford", "PionierL1", "Restena", "Reuna", "Sago",
    "Sanet", "Twaren", "Uran", "Vinaren", "VisionNet", "Zamren",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch only portable Topology Zoo .graph files (no pickle/TM data)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/datasets/topology-zoo-graphs"),
    )
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    repository = Path.cwd().resolve()
    records = []
    for name in TOPOLOGIES:
        destination = args.output / f"{name}.graph"
        url = (
            "https://raw.githubusercontent.com/confiwent/"
            f"Topology_Zoo_dataset/{COMMIT}/{name}/{name}.graph"
        )
        if not destination.is_file():
            if args.offline:
                raise FileNotFoundError(destination)
            last_error: Exception | None = None
            for attempt in range(3):
                request = urllib.request.Request(url, headers={"User-Agent": "TriSched/1.0"})
                try:
                    with urllib.request.urlopen(request, timeout=120) as response:
                        content = response.read()
                    temporary = destination.with_suffix(".graph.part")
                    temporary.write_bytes(content)
                    temporary.replace(destination)
                    last_error = None
                    break
                except Exception as error:  # network errors differ by platform
                    last_error = error
                    time.sleep(1 + attempt)
            if last_error is not None:
                raise last_error
        try:
            portable_path = destination.resolve().relative_to(repository).as_posix()
        except ValueError:
            portable_path = f"external-sha256://{_sha256(destination)}/{destination.name}"
        records.append(
            {
                "name": name,
                "path": portable_path,
                "bytes": destination.stat().st_size,
                "sha256": _sha256(destination),
                "url": url,
            }
        )
    manifest = {
        "format_version": 1,
        "source": "https://github.com/confiwent/Topology_Zoo_dataset",
        "license": "MIT",
        "commit": COMMIT,
        "scope": "topology .graph files only; traffic matrices and pickle samples excluded",
        "files": records,
    }
    manifest_path = args.output / "source_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(manifest_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
