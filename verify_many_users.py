"""The Outside Judge for C3: runs the five experiments once and checks what must hold.

    python verify_many_users.py          # live vLLM through the tunnel (real numbers)
    python verify_many_users.py --mock   # fake server, checks the logic without a GPU

The checks are about behavior (together beats one by one, p95 time to first token grows
with the load, short answers overtake long ones that started earlier, KV-cache usage
climbs with the load, a shared prefix starts faster than a unique one), not about exact
numbers. Times vary from run to run; the JSON keeps the raw values.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from vllm_lab.client import BASE_URL

EXPERIMENTS = [
    "01_one_by_one_vs_together",
    "02_load_sweep",
    "03_continuous_batching",
    "04_paged_kv_cache",
    "05_prefix_caching",
]


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def server_info(base_url: str) -> tuple[str, str]:
    root = base_url.rsplit("/v1", 1)[0]
    try:
        with urllib.request.urlopen(root + "/version", timeout=30) as r:
            version = json.loads(r.read()).get("version", "?")
    except (OSError, ValueError):
        version = "mock"
    try:
        with urllib.request.urlopen(base_url + "/models", timeout=60) as r:
            model = json.loads(r.read())["data"][0]
        weights = f"{model.get('root')} served as '{model['id']}'"
    except (OSError, ValueError, KeyError, IndexError):
        weights = "?"
    return version, weights


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mock", action="store_true")
    p.add_argument("--base-url", default=BASE_URL)
    p.add_argument("--verbose", action="store_true", help="print each experiment's raw result")
    args = p.parse_args()

    from vllm_lab import cli

    base_url = cli.resolve(args)
    t_start = time.perf_counter()
    res = {name: load(name).run(base_url) for name in EXPERIMENTS}
    elapsed = time.perf_counter() - t_start
    r1, r2, r3, r4, r5 = (res[n] for n in EXPERIMENTS)
    if args.verbose:
        for name, value in res.items():
            cli.show(name, value)

    lo, hi = r2["rows"][0], r2["rows"][-1]
    k_lo, k_hi = r4["rows"][0], r4["rows"][-1]
    sh, un = r5["shared_prefix"], r5["unique_prefix"]
    knee = r2["knee_concurrency"]
    rows = [
        ("01. One by one vs together",
         (f"{r1['requests']} req: {r1['one_by_one']['wall_s']:.2f} s -> "
          f"{r1['together']['wall_s']:.2f} s"),
         f"{r1['throughput_gain_x']:.1f}x tok/s",
         r1["together"]["output_tokens_per_s"] > r1["one_by_one"]["output_tokens_per_s"]),
        ("02. Load sweep 1 -> 64 users",
         f"p95 TTFT {lo['p95_ttft_ms']:.0f} -> {hi['p95_ttft_ms']:.0f} ms",
         f"{r2['throughput_gain_x']:.1f}x tok/s, knee {knee if knee else '-'}",
         hi["p95_ttft_ms"] > lo["p95_ttft_ms"]
         and hi["output_tokens_per_s"] > lo["output_tokens_per_s"]),
        ("03. Continuous batching",
         f"short {r3['p50_short_latency_s']:.2f} s, long {r3['p50_long_latency_s']:.2f} s",
         f"{r3['short_finished_first']}/{r3['short_vs_earlier_long_pairs']} overtakes",
         r3["all_short_first"]),
        ("04. Paged KV-cache under load",
         f"KV {k_lo['peak_kv_cache_usage_pct']:.0f}% -> {k_hi['peak_kv_cache_usage_pct']:.0f}%",
         (f"waiting {max(r['peak_waiting'] for r in r4['rows'])}, "
          f"preempt {sum(r['preemptions'] for r in r4['rows'])}"),
         r4["metrics_found"]["kv_cache_usage"] and r4["kv_usage_rises"]),
        ("05. Prefix caching",
         f"TTFT shared {sh['p50_ttft_ms']:.0f} vs unique {un['p50_ttft_ms']:.0f} ms",
         f"hit rate {sh['hit_rate_pct'] if sh['hit_rate_pct'] is not None else '?'}%",
         sh["p50_ttft_ms"] < un["p50_ttft_ms"]),
    ]

    mode = "mock (fake server, numbers are not real)" if args.mock else "live"
    version, weights = server_info(base_url)
    lines = [
        "=" * 104,
        "THE OUTSIDE JUDGE — Many Users, One GPU with vLLM",
        "=" * 104,
        f"Mode    : {mode}",
        (f"Server  : vLLM {version} | model {weights} | KV capacity "
         f"{r4['kv_capacity_tokens'] or '?'} tokens"),
        f"Client  : Python {platform.python_version()} | {platform.system()} {platform.machine()}",
        "-" * 104,
        f"{'Experiment':30} | {'Measured':38} | {'Result':24} | Check",
        "-" * 104,
    ]
    for name, measured, result, ok in rows:
        lines.append(f"{name:30} | {measured:38} | {result:24} | {'PASS' if ok else 'FAIL'}")
    lines += [
        "-" * 104,
        "Times include the network round trip through the tunnel and vary run to run.",
        (f"Load sweep at {hi['concurrency']} users: {hi['output_tokens_per_s']:.0f} tok/s, "
         f"${hi['usd_per_million_output_tokens']:.2f} per 1M output tokens "
         f"(1 user: ${lo['usd_per_million_output_tokens']:.2f})."),
        f"Judge run took {elapsed:.1f} s.",
    ]
    all_ok = all(ok for *_, ok in rows)
    lines.append("ALL CHECKS PASSED." if all_ok else "SOME CHECKS FAILED.")
    report = "\n".join(lines)
    print(report)

    if not args.mock:
        out = ROOT / "benchmarks"
        out.mkdir(exist_ok=True)
        payload = {
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "vllm_version": version,
            "experiments": res,
            "checks": {name: bool(ok) for name, *_, ok in rows},
        }
        (out / "latest_results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        (out / "terminal_output.txt").write_text(report + "\n")
        print("Saved benchmarks/latest_results.json and benchmarks/terminal_output.txt")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
