#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python -m compileall -q unified_transport *.py
python -m unittest discover -s tests -v
python - <<'PY'
import json
from pathlib import Path
expected = {
    "generation/step-1000000-cifar10-generation-only-lowfreq-ot-ce-5k-model-1000000/metrics.json",
    "quality/quality-metrics.json",
    "reverse-margin/metrics.json",
}
base = Path("runs/cifar10-generation-only-lowfreq-ot-ce-1m-evaluation")
missing = [str(base / item) for item in expected if not (base / item).exists()]
if missing:
    print("release metric files not present locally (expected until evaluation is run):")
    print("\n".join(missing))
else:
    generation = json.loads((base / "generation/step-1000000-cifar10-generation-only-lowfreq-ot-ce-5k-model-1000000/metrics.json").read_text())
    quality = json.loads((base / "quality/quality-metrics.json").read_text())
    reverse = json.loads((base / "reverse-margin/metrics.json").read_text())
    assert abs(generation["generated_accuracy"] - 0.8265625238418579) < 1e-6
    assert abs(quality["fid_inception"] - 53.25769034210202) < 1e-6
    assert abs(reverse["results"][0]["accuracy"][-1] - 0.9023) < 1e-6
    print("release metric checks passed")
PY
