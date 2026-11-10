"""02. Load sweep from 1 to 64 users: latency vs throughput, and where the knee is.

At each concurrency level it sends twice as many requests as users (at least 8), every
one with the same prompt length and exactly OUT_TOKENS of answer (`ignore_eos`). It
reports what users feel (p50/p95 time to first token, p50 inter-token latency) and what
the server delivers (aggregate output tokens/s and dollars per million tokens). The knee is
the first level where the time a user waits for a whole answer (p50 latency) grows by a
larger factor than throughput: from there on, adding users mostly adds waiting.

    python experiments/02_load_sweep.py [--mock]
"""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

from vllm_lab import cli, make_client, stream_chat
from vllm_lab.load import Job, run_jobs, summarize

PRICING = Path(__file__).resolve().parent.parent / "benchmarks" / "pricing.json"
LEVELS = (1, 2, 4, 8, 16, 32, 64)
OUT_TOKENS = 32
PROMPT = "Resume en tres frases por qué un envío puede llegar tarde a una tienda en línea."


def find_knee(rows: list[dict]) -> int | None:
    for prev, cur in pairwise(rows):
        tput_gain = cur["output_tokens_per_s"] / max(prev["output_tokens_per_s"], 1e-9)
        latency_gain = cur["p50_latency_ms"] / max(prev["p50_latency_ms"], 1e-9)
        if latency_gain > tput_gain:
            return cur["concurrency"]
    return None


def run(base_url: str, levels: tuple[int, ...] = LEVELS) -> dict:
    per_hour = json.loads(PRICING.read_text())["usd_per_hour"]
    stream_chat(make_client(base_url), "Hola", max_tokens=8)  # warmup
    rows = []
    for users in levels:
        n = max(2 * users, 8)
        # A different first word per request, so prefix caching does not help this test.
        jobs = [Job(f"Solicitud {users}-{i}. {PROMPT}", OUT_TOKENS) for i in range(n)]
        results, _, wall = run_jobs(base_url, jobs, concurrency=users)
        row = {"concurrency": users, **summarize(results, wall)}
        row["usd_per_million_output_tokens"] = round(
            per_hour / max(row["output_tokens_per_s"] * 3600, 1e-9) * 1e6, 2)
        rows.append(row)
    first, last = rows[0], rows[-1]
    return {
        "output_tokens_per_request": OUT_TOKENS,
        "rows": rows,
        "knee_concurrency": find_knee(rows),
        "throughput_gain_x": round(last["output_tokens_per_s"] /
                                   max(first["output_tokens_per_s"], 1e-9), 1),
        "p95_ttft_growth_x": round(last["p95_ttft_ms"] / max(first["p95_ttft_ms"], 1e-9), 1),
    }


def main() -> None:
    args = cli.parse(__doc__.splitlines()[0])
    result = run(cli.resolve(args))
    print(f"\n{'users':>5} | {'p50 TTFT':>9} | {'p95 TTFT':>9} | {'p50 ITL':>8} | "
          f"{'p50 total':>9} | {'tok/s':>8} | {'$/1M tok':>8}")
    for r in result["rows"]:
        print(f"{r['concurrency']:>5} | {r['p50_ttft_ms']:>6.0f} ms | {r['p95_ttft_ms']:>6.0f} ms"
              f" | {r['p50_itl_ms']:>5.1f} ms | {r['p50_latency_ms']:>6.0f} ms"
              f" | {r['output_tokens_per_s']:>8.0f} | "
              f"{r['usd_per_million_output_tokens']:>8.2f}")
    cli.show("Load sweep", {k: v for k, v in result.items() if k != "rows"})


if __name__ == "__main__":
    main()
