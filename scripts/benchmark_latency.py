#!/usr/bin/env python3
"""Measure p50 and p95 for the text and image paths, cold and warm, and write them down.

The number a user waits for is the turn span; the number that explains it is the sum of the
model and media spans inside it. Both are reported, because "our agent is fast" is only a claim
until it says which part of the turn it measured.

With no API key in the environment the text path runs on the deterministic planner and the image
path on a scripted vision answer, so the run is repeatable on any laptop - and it measures
application overhead, not provider latency. The output file says which of the two it measured, in
`model_backing`, so a number can never be read as the wrong kind. For real provider numbers:

    export CALORAI_TEXT_MODEL_API_KEY=... CALORAI_VISION_MODEL_API_KEY=...
    .venv/bin/python scripts/benchmark_latency.py --image path/to/a/plate.jpg
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import shutil
import statistics
import struct
import subprocess
import sys
import tempfile
import time
import zlib
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from calorai_agent.app import build_planner, build_vision, prepare_runtime
from calorai_agent.config import Settings
from calorai_agent.db import Database
from calorai_agent.domain import MediaRef
from calorai_agent.graph import MealAgent
from calorai_agent.observability import FIELDS_ATTR
from calorai_agent.providers import ImagePayload
from calorai_agent.repository import MealRepository
from calorai_agent.tools import MealTools
from calorai_agent.vision import LocalFileMediaSource, VisionInterpreter

START = datetime(2026, 9, 26, 8, tzinfo=UTC)
# Rotated so no two samples share a message, which would be deduplicated as a redelivery.
TEXT_MESSAGES = (
    "had 2 parathas and chai for breakfast",
    "ate 3 rotis and a bowl of dal",
    "had a banana and coffee",
    "two idlis and chutney for the evening",
)
COMPONENT_SPANS = ("model_request", "media_fetch")
ROOT = Path(__file__).resolve().parent.parent


class ScriptedVisionClient:
    """A vision model that answers instantly, so the offline number is the application's."""

    def observe(self, *, system: str, user: str, image: ImagePayload) -> str:
        return json.dumps(
            {
                "items": [
                    {"name": "biryani", "quantity": "1.5", "confidence": 0.92, "alternative": None},
                    {"name": "curd", "quantity": "0.5", "confidence": 0.85, "alternative": None},
                ],
                "unclear": None,
            }
        )


class SpanCollector(logging.Handler):
    """Collects the durations the application already logs, so nothing is timed twice."""

    def __init__(self) -> None:
        super().__init__()
        self.durations: dict[str, float] = {}
        self.components: dict[str, list[float]] = {}
        self.enabled = False

    def start(self) -> None:
        self.durations = {}
        self.components = {}
        self.enabled = True

    def emit(self, record: logging.LogRecord) -> None:
        if not self.enabled:
            return
        fields = getattr(record, FIELDS_ATTR, None)
        duration = fields.get("duration_ms") if fields else None
        if not isinstance(duration, (int, float)):
            return
        event = record.getMessage()
        if event == "turn":
            self.durations = dict(fields)
        elif event in COMPONENT_SPANS:
            self.components.setdefault(event, []).append(float(duration))


@dataclass
class Samples:
    """One path's measurements: turn milliseconds, and the spans that made them up."""

    turns: list[float] = field(default_factory=list)
    components: dict[str, list[float]] = field(default_factory=dict)
    routes: dict[str, int] = field(default_factory=dict)

    def record(self, durations: dict[str, float], components: dict[str, list[float]]) -> None:
        if "duration_ms" in durations:
            self.turns.append(float(durations["duration_ms"]))
        route = durations.get("route")
        if isinstance(route, str):
            self.routes[route] = self.routes.get(route, 0) + 1
        for event, values in components.items():
            self.components.setdefault(event, []).extend(values)

    def as_json(self) -> dict[str, Any]:
        return {
            "samples": len(self.turns),
            "p50_ms": _percentile(self.turns, 50),
            "p95_ms": _percentile(self.turns, 95),
            "min_ms": round(min(self.turns), 2) if self.turns else None,
            "max_ms": round(max(self.turns), 2) if self.turns else None,
            "mean_ms": round(statistics.fmean(self.turns), 2) if self.turns else None,
            "routes": self.routes,
            "components": {
                event: {"p50_ms": _percentile(values, 50), "p95_ms": _percentile(values, 95)}
                for event, values in self.components.items()
            },
        }


def _percentile(values: list[float], percent: int) -> float | None:
    """Nearest-rank percentile: a real sample's value, never an interpolation between two."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, -(-percent * len(ordered) // 100) - 1))
    return round(ordered[rank], 2)


def png_bytes(size: int = 8) -> bytes:
    """A valid tiny PNG: real enough for a provider to open, small enough to ignore."""
    rows = b"".join(b"\x00" + bytes((row * 31 % 256,) * size) for row in range(size))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", size, size, 8, 0, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


class Benchmark:
    """Builds a fresh application per measurement and times the turn the user would wait for."""

    def __init__(self, settings: Settings, image: Path | None, samples: int, warm_up: int) -> None:
        self.settings = settings
        self.image = image
        self.samples = samples
        self.warm_up = warm_up
        self.workdir = Path(tempfile.mkdtemp(prefix="calorai-bench-"))
        self.planner = build_planner(settings)
        live_vision = build_vision(settings)
        self.vision = live_vision or VisionInterpreter(
            ScriptedVisionClient(), LocalFileMediaSource(), model="scripted-stand-in"
        )
        self.live_vision = live_vision is not None
        self.collector = SpanCollector()
        logger = logging.getLogger("calorai_agent")
        logger.handlers = [self.collector]
        logger.level = logging.DEBUG
        logger.propagate = False

    def close(self) -> None:
        logging.getLogger("calorai_agent").handlers = []
        shutil.rmtree(self.workdir, ignore_errors=True)

    def application(self, tag: str) -> MealAgent:
        database = Database(self.workdir / f"{tag}.sqlite3")
        database.initialize()
        repository = MealRepository(database)
        repository.ensure_user("bench-user", "UTC")
        return MealAgent(self.planner, MealTools(repository), vision=self.vision)

    def photo(self, name: str) -> MediaRef:
        path = self.workdir / f"{name}.png"
        path.write_bytes(self.image.read_bytes() if self.image else png_bytes())
        return MediaRef(external_id=name, locator=str(path))

    def turn(self, agent: MealAgent, index: int, photo: bool) -> None:
        moment = START + timedelta(minutes=index)
        media = self.photo(f"bench-{index}") if photo else None
        text = "" if photo else TEXT_MESSAGES[index % len(TEXT_MESSAGES)]
        self.collector.start()
        started = time.perf_counter()
        agent.invoke("bench-user", text, timezone="UTC", now=moment, media=media)
        if not self.collector.durations:
            # A turn that logged nothing still cost the user this much.
            self.collector.durations = {"duration_ms": (time.perf_counter() - started) * 1000}

    def measure(self, name: str, *, photo: bool, cold: bool) -> Samples:
        """Cold pays for a database nobody has opened yet; warm pays for the model."""
        result = Samples()
        if cold:
            for index in range(self.samples):
                self.turn(self.application(f"cold-{name}-{index}"), index, photo)
                result.record(self.collector.durations, self.collector.components)
            return result
        agent = self.application(f"warm-{name}")
        for index in range(self.warm_up + self.samples):
            self.turn(agent, index, photo)
            if index >= self.warm_up:
                result.record(self.collector.durations, self.collector.components)
        return result

    def run(self) -> dict[str, Any]:
        paths = {
            "text_cold": self.measure("text", photo=False, cold=True),
            "text_warm": self.measure("text", photo=False, cold=False),
            "image_cold": self.measure("image", photo=True, cold=True),
            "image_warm": self.measure("image", photo=True, cold=False),
        }
        return {
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "command": " ".join(sys.argv[1:]) or "python scripts/benchmark_latency.py",
            "samples": self.samples,
            "warm_up": self.warm_up,
            "environment": self.environment(),
            "results": {name: path.as_json() for name, path in paths.items()},
        }

    def environment(self) -> dict[str, Any]:
        live_text = type(self.planner).__name__ == "ModelPlanner"
        return {
            "python": platform.python_version(),
            "platform": f"{platform.system()} {platform.release()}",
            "machine": platform.machine(),
            "cpus": os.cpu_count() or 1,
            "git_commit": _git_commit(),
            "database": "sqlite",
            "text_model": self.settings.text_model if live_text else "rule-based planner",
            "vision_model": self.vision.model,
            "model_backing": {
                "text": "live provider" if live_text else "deterministic stand-in (no API key)",
                "vision": "live provider" if self.live_vision else "scripted stand-in (no API key)",
            },
            "note": (
                "Stand-in numbers are application overhead only: no provider round trip is "
                "included. Re-run with API keys for user-visible latency."
            ),
        }


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except OSError:  # pragma: no cover - a reviewer without git still gets a benchmark
        return None


def report(payload: dict[str, Any]) -> None:
    backing = payload["environment"]["model_backing"]
    print(
        f"n={payload['samples']} warm-up={payload['warm_up']} "
        f"text={backing['text']} vision={backing['vision']}"
    )
    print(f"{'path':<12} {'p50 ms':>9} {'p95 ms':>9} {'mean':>8} {'min':>8} {'max':>9}")
    for name, result in payload["results"].items():
        print(
            f"{name:<12} {result['p50_ms']:>9} {result['p95_ms']:>9} "
            f"{result['mean_ms']:>8} {result['min_ms']:>8} {result['max_ms']:>9}"
        )
        for event, values in result["components"].items():
            print(f"{'':<12}   - {event} p50={values['p50_ms']} p95={values['p95_ms']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark the text and image turn paths")
    parser.add_argument("--samples", type=int, default=40, help="turns measured per path")
    parser.add_argument("--warm-up", type=int, default=5, help="turns discarded on the warm paths")
    parser.add_argument(
        "--image", type=Path, help="a real plate photo for the image path (default: tiny PNG)"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "benchmarks" / "latency.json",
        help="where the machine-readable result is written",
    )
    args = parser.parse_args()

    settings = prepare_runtime(Settings.from_env())
    bench = Benchmark(settings, args.image, args.samples, max(0, args.warm_up))
    try:
        payload = bench.run()
    finally:
        bench.close()

    report(payload)
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
