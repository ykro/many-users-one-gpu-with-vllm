"""Runs every experiment against the built-in mock server (no GPU needed)."""

import importlib.util
from pathlib import Path

import pytest

from vllm_lab import make_client, metrics, mock_server, stream_chat
from vllm_lab.load import Job, percentile, run_jobs

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def base_url():
    server, url = mock_server.start()
    yield url
    server.shutdown()


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_percentile_nearest_rank():
    values = [float(v) for v in range(1, 21)]
    assert percentile(values, 50) == 10 and percentile(values, 95) == 19
    assert percentile([], 95) == 0.0


def test_metrics_parser_tolerates_names_and_labels():
    text = (
        "# TYPE vllm:gpu_cache_usage_perc gauge\n"
        'vllm:gpu_cache_usage_perc{model_name="gemma",engine="0"} 0.25\n'
        'vllm:num_preemptions_total{model_name="gemma"} 3.0\n'
        'vllm:num_preemptions_created{model_name="gemma"} 1.7e9\n'
        'vllm:cache_config_info{block_size="16",num_gpu_blocks="1000"} 1.0\n'
    )
    rows = metrics.parse(text)
    assert metrics.pick(rows, metrics.KV_USAGE, agg="max") == 0.25
    assert metrics.pick(rows, metrics.PREEMPTIONS) == 3.0  # the _created series is skipped
    assert metrics.pick(rows, metrics.WAITING) is None
    assert metrics.kv_capacity_tokens(rows) == 16000


def test_stream_counts_every_token(base_url):
    t = stream_chat(make_client(base_url), "hola", max_tokens=12, ignore_eos=True)
    assert t.chunks == t.completion_tokens == 12
    assert 0 < t.ttft_s < t.total_s


def test_mock_queues_requests_when_kv_is_full():
    server, url = mock_server.start(num_blocks=16)  # 256 tokens of KV-cache
    try:
        jobs = [Job(f"pedido {i} " * 10, 64) for i in range(8)]
        with metrics.Sampler(url, every_s=0.01) as sampler:
            results, _, _ = run_jobs(url, jobs, concurrency=8)
        assert all(r.completion_tokens == 64 for r in results)
        assert sampler.peak("requests_waiting") > 0
        assert sampler.peak("kv_cache_usage") > 0.8
    finally:
        server.shutdown()


def test_01_together_beats_one_by_one(base_url):
    r = load("01_one_by_one_vs_together").run(base_url)
    assert r["together"]["output_tokens_per_s"] > r["one_by_one"]["output_tokens_per_s"]


def test_02_latency_grows_with_load(base_url):
    r = load("02_load_sweep").run(base_url, levels=(1, 8, 64))
    lo, hi = r["rows"][0], r["rows"][-1]
    assert hi["p95_ttft_ms"] > lo["p95_ttft_ms"]
    assert hi["output_tokens_per_s"] > lo["output_tokens_per_s"]


def test_03_short_requests_overtake_long_ones(base_url):
    r = load("03_continuous_batching").run(base_url)
    assert r["all_short_first"]
    assert r["p50_short_latency_s"] < r["static_batching_p50_short_latency_s"]


def test_04_kv_usage_rises_and_requests_wait(base_url):
    r = load("04_paged_kv_cache").run(base_url, levels=(1, 64))
    assert r["kv_usage_rises"] and r["requests_waited"]
    assert r["kv_capacity_tokens"] == mock_server.NUM_BLOCKS * mock_server.BLOCK_SIZE


def test_05_shared_prefix_starts_faster(base_url):
    r = load("05_prefix_caching").run(base_url)
    assert r["shared_prefix"]["p50_ttft_ms"] < r["unique_prefix"]["p50_ttft_ms"]
    assert r["shared_prefix"]["hit_rate_pct"] > 50 > r["unique_prefix"]["hit_rate_pct"]
