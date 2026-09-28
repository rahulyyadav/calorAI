"""Phase 6d: the benchmark's own arithmetic.

A p95 is a claim about a distribution, so the two things that can silently misreport it — the
percentile rank and the photo the image path is timed against — are tested here. The latency
numbers themselves live in `benchmarks/latency.json`.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from calorai_agent.domain import MediaRef
from calorai_agent.vision import LocalFileMediaSource

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def benchmark() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "benchmark_latency", ROOT / "scripts" / "benchmark_latency.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_p95_is_a_real_sample_and_never_an_invented_number(benchmark: ModuleType) -> None:
    values = [float(value) for value in range(1, 41)]

    assert benchmark._percentile(values, 50) == 20.0
    assert benchmark._percentile(values, 95) == 38.0
    assert benchmark._percentile([*values, 9_000.0], 95) == 39.0


def test_a_path_that_measured_nothing_reports_nothing(benchmark: ModuleType) -> None:
    reported = benchmark.Samples().as_json()

    assert reported["samples"] == 0
    assert reported["p50_ms"] is None
    assert reported["p95_ms"] is None
    assert "setup_p50_ms" not in reported


def test_a_read_path_records_that_it_billed_nothing(benchmark: ModuleType) -> None:
    """The count of model requests per sample is the fast path's evidence, not its prose.

    A number can only be compared to another number if the report says what was paid for it, so the
    benchmark carries the billing count next to the milliseconds.
    """
    billed = benchmark.Samples()
    billed.record({"duration_ms": 10.0}, {"model_request": [800.0]})
    free = benchmark.Samples()
    free.record({"duration_ms": 4.0, "route": "get_totals"}, {})
    cold = benchmark.Samples()
    cold.turns.append(9.0)
    cold.setup.extend([20.0, 30.0])

    assert billed.as_json()["model_calls_per_sample"] == 1.0
    assert free.as_json()["model_calls_per_sample"] == 0.0
    assert free.as_json()["routes"] == {"get_totals": 1}
    assert cold.as_json()["setup_p50_ms"] == 20.0
    assert cold.as_json()["setup_p95_ms"] == 30.0


def test_the_sample_photo_is_one_the_media_loader_accepts(
    benchmark: ModuleType, tmp_path: Path
) -> None:
    path = tmp_path / "plate.png"
    path.write_bytes(benchmark.png_bytes())

    payload = LocalFileMediaSource().fetch(MediaRef(external_id="bench", locator=str(path)))

    assert payload.mime_type == "image/png"
    assert payload.data.startswith(b"\x89PNG")


def test_the_benchmark_result_file_is_machine_readable_and_labels_its_backing() -> None:
    report = json.loads((ROOT / "benchmarks" / "latency.json").read_text())

    assert {"environment", "results", "samples", "definitions"} <= set(report)
    for name in (
        "text_cold",
        "text_warm",
        "read_cold",
        "read_warm",
        "image_cold",
        "image_warm",
    ):
        result = report["results"][name]
        assert result["samples"] == report["samples"]
        assert result["p50_ms"] is not None and result["p95_ms"] >= result["p50_ms"]
    # A read of the record is computed from the database, so it bills nothing however the
    # deployment is configured — that is the claim, and the artifact has to keep saying it.
    for name in ("read_cold", "read_warm"):
        assert report["results"][name]["model_calls_per_sample"] == 0
    # "cold" would be meaningless without saying where schema setup was measured.
    assert "setup_p50_ms" in report["definitions"]["cold"]
    for name in ("text_cold", "read_cold", "image_cold"):
        assert report["results"][name]["setup_p50_ms"] is not None
    # A number that does not say whether a provider was in the loop is not evidence.
    assert set(report["environment"]["model_backing"]) == {"text", "vision"}
    assert all(report["environment"]["model_backing"].values())
