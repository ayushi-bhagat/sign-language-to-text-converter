"""Performance benchmark script measuring FPS, model load time, and inference latency against configured targets.

Usage:
    .venv\\Scripts\\python.exe scripts\\benchmark.py
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys
import time
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from sign_language.config.loader import get_config
from sign_language.features.extractor import LandmarkExtractor
from sign_language.inference.predictor import SignPredictor
from sign_language.inference.stabilizer import PredictionStabilizer
from sign_language.utils.logging_setup import configure_logging


def benchmark() -> None:
    configure_logging()
    logger = logging.getLogger("benchmark")
    cfg = get_config()

    print("\n" + "=" * 65)
    print("  PERFORMANCE BENCHMARK")
    print("=" * 65)
    print(f"  Target Min FPS             : {cfg.performance.targets.min_fps:.1f}")
    print(f"  Target Max Inf Latency     : {cfg.performance.targets.max_inference_latency_ms:.1f} ms")
    print(f"  Target Max Model Load Time : {cfg.performance.targets.max_model_load_ms:.1f} ms")
    print("-" * 65)

    # 1. Measure Model Load Time
    t0 = time.perf_counter()
    predictor = SignPredictor(cfg)
    try:
        predictor.load()
        load_time_ms = (time.perf_counter() - t0) * 1000.0
        load_pass = load_time_ms <= cfg.performance.targets.max_model_load_ms
        status_load = "✅ PASS" if load_pass else "❌ FAIL"
        print(f"  Model Load Time            : {load_time_ms:6.2f} ms  {status_load}")
    except Exception as exc:
        print(f"  Model Load Time            : FAILED ({exc})")
        return

    # 2. Measure Feature Extraction + Predictor Latency (100 synthetic iterations)
    latencies = []
    dummy_vec = np.random.randn(63).astype(np.float32)

    for _ in range(100):
        t_start = time.perf_counter()
        pred = predictor.predict(dummy_vec)
        t_end = time.perf_counter()
        latencies.append((t_end - t_start) * 1000.0)

    p50_latency = float(np.percentile(latencies, 50))
    p95_latency = float(np.percentile(latencies, 95))
    lat_pass = p95_latency <= cfg.performance.targets.max_inference_latency_ms
    status_lat = "✅ PASS" if lat_pass else "❌ FAIL"
    print(f"  Inference Latency (p50)    : {p50_latency:6.2f} ms")
    print(f"  Inference Latency (p95)    : {p95_latency:6.2f} ms  {status_lat}")

    # 3. Stabilizer Benchmark
    stabilizer = PredictionStabilizer(cfg.inference)
    stab_times = []
    for i in range(100):
        t_start = time.perf_counter()
        stabilizer.update(pred, now=time.monotonic())
        t_end = time.perf_counter()
        stab_times.append((t_end - t_start) * 1000.0)

    p95_stab = float(np.percentile(stab_times, 95))
    print(f"  Stabilizer Latency (p95)   : {p95_stab:6.3f} ms")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    benchmark()
