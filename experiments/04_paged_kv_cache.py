"""04. Paged KV-cache: watch the memory fill up and the queue appear, from /metrics.

Every request keeps its KV-cache (what the model remembers of the prompt and of what it
already wrote) in GPU memory. vLLM's PagedAttention manages that memory the way an
operating system manages RAM with virtual memory: in fixed-size blocks handed out as the
context grows, from wherever there is a free block, instead of reserving one contiguous
region for the longest answer a request could ever write. Nothing is wasted on space a
request may never use, so many more requests fit at once.

This experiment ramps the number of users and polls vLLM's live /metrics meanwhile:
KV-cache usage climbs with the load, and once the blocks run out new requests wait
(`num_requests_waiting` > 0) or running ones get preempted. If the server exposes its
KV capacity, the answer length is sized so the top level asks for more than fits.

    python experiments/04_paged_kv_cache.py [--mock] [--levels 1,8,32,64]
"""

from __future__ import annotations

from vllm_lab import cli, make_client, metrics, stream_chat
from vllm_lab.load import Job, run_jobs

LEVELS = (1, 8, 32, 64)
PARAGRAPH = ("La tienda registra cada pedido con su cliente, dirección, productos, peso y "
             "fecha prometida de entrega. ")
PROMPT = PARAGRAPH * 4 + "Escribe un reporte del estado de este pedido."
DEFAULT_OUT_TOKENS = 64
MAX_OUT_TOKENS = 768
OVERSUBSCRIBE = 1.25  # the top level asks for 25 % more KV-cache than exists


def answer_length(capacity_tokens: int | None, top_users: int, prompt_tokens: int) -> int:
    if not capacity_tokens:
        return DEFAULT_OUT_TOKENS
    wanted = int(capacity_tokens * OVERSUBSCRIBE / top_users) - prompt_tokens
    return max(DEFAULT_OUT_TOKENS, min(MAX_OUT_TOKENS, wanted))


def run(base_url: str, levels: tuple[int, ...] = LEVELS) -> dict:
    warm = stream_chat(make_client(base_url), PROMPT, max_tokens=8)  # warmup + prompt size
    rows0 = metrics.fetch(base_url)
    capacity = metrics.kv_capacity_tokens(rows0)
    found = {name: metrics.pick(rows0, prefixes) is not None for name, prefixes in (
        ("kv_cache_usage", metrics.KV_USAGE), ("requests_running", metrics.RUNNING),
        ("requests_waiting", metrics.WAITING), ("preemptions", metrics.PREEMPTIONS))}
    out_tokens = answer_length(capacity, max(levels), warm.prompt_tokens)
    rows = []
    for users in levels:
        before = metrics.snapshot(base_url)
        jobs = [Job(f"Pedido {users}-{i}. {PROMPT}", out_tokens) for i in range(users)]
        with metrics.Sampler(base_url) as sampler:
            _, _, wall = run_jobs(base_url, jobs, concurrency=users)
        after = metrics.snapshot(base_url)
        rows.append({
            "users": users,
            "kv_tokens_requested": users * (warm.prompt_tokens + out_tokens),
            "peak_kv_cache_usage_pct": round(100 * sampler.peak("kv_cache_usage"), 1),
            "peak_running": int(sampler.peak("requests_running")),
            "peak_waiting": int(sampler.peak("requests_waiting")),
            "preemptions": int((after["preemptions_total"] or 0) -
                               (before["preemptions_total"] or 0)),
            "wall_s": round(wall, 2),
            "samples": len(sampler.samples),
        })
    return {
        "metrics_found": found,
        "kv_capacity_tokens": capacity,
        "prompt_tokens": warm.prompt_tokens,
        "output_tokens_per_request": out_tokens,
        "rows": rows,
        "kv_usage_rises": rows[-1]["peak_kv_cache_usage_pct"] > rows[0]["peak_kv_cache_usage_pct"],
        "requests_waited": any(r["peak_waiting"] > 0 or r["preemptions"] > 0 for r in rows),
    }


def main() -> None:
    p = cli.parser(__doc__.splitlines()[0])
    p.add_argument("--levels", default=",".join(map(str, LEVELS)),
                   help="comma-separated user counts")
    args = p.parse_args()
    levels = tuple(int(x) for x in args.levels.split(","))
    result = run(cli.resolve(args), levels)
    print(f"\n{'users':>5} | {'KV asked':>9} | {'peak KV %':>9} | {'running':>7} | "
          f"{'waiting':>7} | {'preempt':>7}")
    for r in result["rows"]:
        print(f"{r['users']:>5} | {r['kv_tokens_requested']:>9} | "
              f"{r['peak_kv_cache_usage_pct']:>9.1f} | {r['peak_running']:>7} | "
              f"{r['peak_waiting']:>7} | {r['preemptions']:>7}")
    cli.show("Paged KV-cache", {k: v for k, v in result.items() if k != "rows"})


if __name__ == "__main__":
    main()
