import argparse
import io
import json
import statistics
import time
import wave

import requests


def generate_sample(sample_rate: int = 16000, duration: float = 0.5) -> bytes:
    import math
    import struct

    frequency = 440.0
    frames = bytearray()
    for index in range(int(sample_rate * duration)):
        value = int(12000 * math.sin(2 * math.pi * frequency * index / sample_rate))
        frames.extend(struct.pack("<h", value))
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(frames)
    return buffer.getvalue()


def percentile(values: list[float], percentile_value: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile_value / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure server and client API latency")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--sample")
    args = parser.parse_args()
    if args.requests < 1 or args.warmup < 0:
        parser.error("--requests must be positive and --warmup cannot be negative")

    if args.sample:
        with open(args.sample, "rb") as sample_file:
            payload = sample_file.read()
    else:
        payload = generate_sample()
    server_latencies: list[float] = []
    client_latencies: list[float] = []
    session = requests.Session()
    for index in range(args.warmup + args.requests):
        started_at = time.perf_counter()
        response = session.post(
            f"{args.url.rstrip('/')}/predict",
            files={"file": ("benchmark.wav", payload, "audio/wav")},
            timeout=30,
        )
        client_ms = (time.perf_counter() - started_at) * 1000
        response.raise_for_status()
        server_ms = float(response.json()["latency_ms"])
        if index >= args.warmup:
            server_latencies.append(server_ms)
            client_latencies.append(client_ms)

    result = {}
    for name, values in (
        ("server", server_latencies),
        ("client_end_to_end", client_latencies),
    ):
        result[name] = {
            "mean_ms": round(statistics.fmean(values), 2),
            "p50_ms": round(percentile(values, 50), 2),
            "p95_ms": round(percentile(values, 95), 2),
        }
    result["requests"] = args.requests
    result["warmup_requests"] = args.warmup
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
