"""Reads vLLM's Prometheus endpoint (/metrics) without any Prometheus library.

Metric names move between vLLM versions (`vllm:gpu_cache_usage_perc` became
`vllm:kv_cache_usage_perc`, counters gain a `_total` suffix), so every lookup takes a list
of name prefixes and uses the first one that exists.
"""

from __future__ import annotations

import threading
import time
import urllib.request
from typing import Self

KV_USAGE = ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc", "vllm:kv_cache_usage")
RUNNING = ("vllm:num_requests_running",)
WAITING = ("vllm:num_requests_waiting",)
PREEMPTIONS = ("vllm:num_preemptions",)
PREFIX_HITS = ("vllm:prefix_cache_hits", "vllm:gpu_prefix_cache_hits")
PREFIX_QUERIES = ("vllm:prefix_cache_queries", "vllm:gpu_prefix_cache_queries")
CACHE_CONFIG = ("vllm:cache_config_info",)

# Series that Prometheus clients add next to the real value; never what we want.
_SKIP_SUFFIXES = ("_created", "_bucket", "_sum", "_count")


def root_of(base_url: str) -> str:
    return base_url.rsplit("/v1", 1)[0]


def parse(text: str) -> list[tuple[str, dict, float]]:
    """Prometheus text format -> [(name, labels, value)]."""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        labels: dict[str, str] = {}
        if "{" in line:
            name = line[: line.index("{")]
            inner = line[line.index("{") + 1 : line.rindex("}")]
            rest = line[line.rindex("}") + 1 :].split()
            for part in inner.split('",'):
                if "=" in part:
                    k, v = part.split("=", 1)
                    labels[k.strip().strip(",")] = v.strip().strip('"')
        else:
            name, *rest = line.split()
        try:
            rows.append((name, labels, float(rest[0])))
        except (IndexError, ValueError):
            continue
    return rows


def fetch(base_url: str, timeout: float = 30) -> list[tuple[str, dict, float]]:
    with urllib.request.urlopen(root_of(base_url) + "/metrics", timeout=timeout) as resp:
        return parse(resp.read().decode())


def pick(rows: list[tuple[str, dict, float]], prefixes: tuple[str, ...],
         agg: str = "sum") -> float | None:
    """Value of the first metric matching one of the prefixes, summed (or max) over labels."""
    for prefix in prefixes:
        values = [v for n, _, v in rows if n.startswith(prefix)
                  and not n.endswith(_SKIP_SUFFIXES)]
        if values:
            return max(values) if agg == "max" else sum(values)
    return None


def kv_capacity_tokens(rows: list[tuple[str, dict, float]]) -> int | None:
    """How many tokens fit in the KV-cache, as the server itself reports it.

    vLLM 0.33 publishes `kv_cache_size_tokens` (the same number as "GPU KV cache size" in its
    startup log). num_gpu_blocks x block_size is only the fallback: with Gemma 4's hybrid
    cache it overstates the capacity by about 2.4x (225,024 against 93,138 on an L4).
    """
    for name, labels, _ in rows:
        if name.startswith(CACHE_CONFIG):
            published = labels.get("kv_cache_size_tokens", "")
            if published.isdigit():
                return int(published)
            try:
                return int(labels["num_gpu_blocks"]) * int(labels["block_size"])
            except (KeyError, ValueError):
                return None
    return None


def snapshot(base_url: str) -> dict:
    rows = fetch(base_url)
    return {
        "kv_cache_usage": pick(rows, KV_USAGE, agg="max"),
        "requests_running": pick(rows, RUNNING),
        "requests_waiting": pick(rows, WAITING),
        "preemptions_total": pick(rows, PREEMPTIONS),
        "prefix_cache_hits": pick(rows, PREFIX_HITS),
        "prefix_cache_queries": pick(rows, PREFIX_QUERIES),
    }


class Sampler:
    """Polls /metrics in a background thread while a load test runs."""

    def __init__(self, base_url: str, every_s: float = 0.05):
        self.base_url, self.every_s = base_url, every_s
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._t0 = 0.0

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                snap = snapshot(self.base_url)
                snap["t_s"] = round(time.perf_counter() - self._t0, 3)
                self.samples.append(snap)
            except OSError:
                pass
            self._stop.wait(self.every_s)

    def __enter__(self) -> Self:
        self._t0 = time.perf_counter()
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=30)

    def peak(self, key: str) -> float:
        values = [s[key] for s in self.samples if s.get(key) is not None]
        return max(values) if values else 0.0
