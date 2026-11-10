"""Load generation: fire many streamed requests at once and summarize what users felt."""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass

from vllm_lab.client import StreamTiming, astream_chat, make_async_client


@dataclass
class Job:
    """One request: what to ask, how long the answer must be, and when to send it."""

    prompt: str
    max_tokens: int
    system: str | None = None
    delay_s: float = 0.0  # seconds after the start of the run
    tag: str = ""


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile (p in 0..100). Plain and easy to check by hand."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100 * len(ordered)))
    return ordered[rank - 1]


async def _fire(base_url: str, jobs: list[Job], concurrency: int) -> tuple[list[StreamTiming],
                                                                            float, float]:
    client = make_async_client(base_url)
    gate = asyncio.Semaphore(concurrency)
    t0 = time.perf_counter()

    async def one(job: Job) -> StreamTiming:
        if job.delay_s:
            await asyncio.sleep(max(0.0, t0 + job.delay_s - time.perf_counter()))
        async with gate:
            return await astream_chat(client, job.prompt, max_tokens=job.max_tokens,
                                      system=job.system, ignore_eos=True)

    try:
        results = await asyncio.gather(*(one(j) for j in jobs))
    finally:
        await client.close()
    return list(results), t0, time.perf_counter() - t0


def run_jobs(base_url: str, jobs: list[Job], concurrency: int) -> tuple[list[StreamTiming],
                                                                        float, float]:
    """Runs the jobs with at most `concurrency` in flight.

    Returns (timings in job order, t0 as perf_counter, wall seconds).
    """
    return asyncio.run(_fire(base_url, jobs, concurrency))


def summarize(results: list[StreamTiming], wall_s: float) -> dict:
    ttft = [r.ttft_s for r in results]
    latency = [r.total_s for r in results]
    itl = [r.itl_s for r in results]
    out_tokens = sum(r.completion_tokens for r in results)
    return {
        "requests": len(results),
        "wall_s": round(wall_s, 3),
        "p50_ttft_ms": round(percentile(ttft, 50) * 1000, 1),
        "p95_ttft_ms": round(percentile(ttft, 95) * 1000, 1),
        "p50_itl_ms": round(percentile(itl, 50) * 1000, 2),
        "p50_latency_ms": round(percentile(latency, 50) * 1000, 1),
        "output_tokens": out_tokens,
        "output_tokens_per_s": round(out_tokens / max(wall_s, 1e-9), 1),
    }
