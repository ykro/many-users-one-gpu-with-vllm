"""01. One user leaves the GPU mostly idle: the same requests, one by one vs together.

Sends N requests with the same answer length twice. First one after another, as a script
with a for-loop would; then all at once. Each answer takes about the same time in both
runs, but together they finish in a fraction of the wall time, so the aggregate output
tokens per second multiplies. Serving one user uses a sliver of what the GPU can do.

    python experiments/01_one_by_one_vs_together.py          # live server (tunnel open)
    python experiments/01_one_by_one_vs_together.py --mock   # no GPU
"""

from __future__ import annotations

import time

from vllm_lab import cli, make_client, stream_chat
from vllm_lab.load import Job, percentile, run_jobs

N_REQUESTS = 8
OUT_TOKENS = 48


def prompts() -> list[str]:
    return [f"Solicitud {i}. Cuenta una historia corta sobre un envío que llega tarde."
            for i in range(N_REQUESTS)]


def run(base_url: str) -> dict:
    client = make_client(base_url)
    stream_chat(client, "Hola", max_tokens=8)  # warmup: first request after a cold start

    t0 = time.perf_counter()
    one_by_one = [stream_chat(client, p, max_tokens=OUT_TOKENS, ignore_eos=True)
                  for p in prompts()]
    seq_wall = time.perf_counter() - t0

    jobs = [Job(p, OUT_TOKENS) for p in prompts()]
    together, _, conc_wall = run_jobs(base_url, jobs, concurrency=N_REQUESTS)

    def side(results, wall):
        tokens = sum(r.completion_tokens for r in results)
        return {
            "wall_s": round(wall, 3),
            "output_tokens": tokens,
            "output_tokens_per_s": round(tokens / max(wall, 1e-9), 1),
            "p50_seconds_per_answer": round(percentile([r.total_s for r in results], 50), 3),
            "p50_tokens_per_s_per_user": round(
                percentile([r.decode_tokens_per_s for r in results], 50), 1),
        }

    seq, conc = side(one_by_one, seq_wall), side(together, conc_wall)
    return {
        "requests": N_REQUESTS,
        "output_tokens_per_request": OUT_TOKENS,
        "one_by_one": seq,
        "together": conc,
        "throughput_gain_x": round(conc["output_tokens_per_s"] /
                                   max(seq["output_tokens_per_s"], 1e-9), 2),
        "per_user_slowdown_x": round(seq["p50_tokens_per_s_per_user"] /
                                     max(conc["p50_tokens_per_s_per_user"], 1e-9), 2),
    }


def main() -> None:
    args = cli.parse(__doc__.splitlines()[0])
    cli.show("One by one vs together", run(cli.resolve(args)))


if __name__ == "__main__":
    main()
