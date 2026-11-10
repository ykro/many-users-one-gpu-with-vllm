"""03. Continuous batching: a short answer does not wait for a long one.

Four long answers start first. While they are being written, eight short questions arrive
one by one. With continuous batching each short request joins the running batch at the
next step and leaves as soon as its last token is out, long before the long answers that
started earlier. The run is saved as a timeline (sent, first token, finished, per request)
that a diagram or a video can plot. For contrast it also estimates what a server with
static batching (the batch closes and runs until its longest answer ends) would have done.

    python experiments/03_continuous_batching.py [--mock] [--out timeline.json]

Live runs save benchmarks/timeline_continuous_batching.json unless --out says otherwise.
"""

from __future__ import annotations

import json
from pathlib import Path

from vllm_lab import cli, make_client, stream_chat
from vllm_lab.load import Job, percentile, run_jobs

ROOT = Path(__file__).resolve().parent.parent
LONG_TOKENS = 320        # long enough to still be writing after the last short one ends
SHORT_TOKENS = 16
N_LONG = 4
N_SHORT = 8
LONG_GAP_S = 0.05        # long requests start 50 ms apart
SHORT_FIRST_S = 0.2      # the first short one arrives once the long ones are writing
SHORT_GAP_S = 0.06       # all shorts arrive in under 0.7 s; the longs need over 1.5 s


def jobs() -> list[Job]:
    out = [Job(f"Petición larga {i}. Escribe un informe detallado sobre la logística de una "
               "tienda en línea.", LONG_TOKENS, delay_s=i * LONG_GAP_S, tag=f"long-{i}")
           for i in range(N_LONG)]
    out += [Job(f"Pregunta corta {i}. ¿Qué es un token?", SHORT_TOKENS,
                delay_s=SHORT_FIRST_S + i * SHORT_GAP_S, tag=f"short-{i}")
            for i in range(N_SHORT)]
    return out


def run(base_url: str) -> dict:
    stream_chat(make_client(base_url), "Hola", max_tokens=8)  # warmup
    plan = jobs()
    results, t0, wall = run_jobs(base_url, plan, concurrency=len(plan))
    timeline = []
    for job, r in zip(plan, results, strict=True):
        start = r.sent_at - t0
        timeline.append({
            "id": job.tag, "kind": job.tag.split("-")[0], "max_tokens": job.max_tokens,
            "sent_s": round(start, 4), "first_token_s": round(start + r.ttft_s, 4),
            "finished_s": round(start + r.total_s, 4), "tokens": r.completion_tokens,
        })
    longs = [t for t in timeline if t["kind"] == "long"]
    shorts = [t for t in timeline if t["kind"] == "short"]
    # Every short request must end before every long one that was sent before it.
    pairs = [(s, lg) for s in shorts for lg in longs if lg["sent_s"] < s["sent_s"]]
    overtakes = sum(s["finished_s"] < lg["finished_s"] for s, lg in pairs)
    short_latency = [s["finished_s"] - s["sent_s"] for s in shorts]
    long_end = max(lg["finished_s"] for lg in longs)
    short_alone = percentile(short_latency, 50)
    static_latency = [long_end - s["sent_s"] + short_alone for s in shorts]
    return {
        "timeline": timeline,
        "wall_s": round(wall, 3),
        "short_vs_earlier_long_pairs": len(pairs),
        "short_finished_first": overtakes,
        "all_short_first": bool(pairs) and overtakes == len(pairs),
        "p50_short_latency_s": round(short_alone, 3),
        "p50_long_latency_s": round(percentile([lg["finished_s"] - lg["sent_s"]
                                                for lg in longs], 50), 3),
        "static_batching_p50_short_latency_s": round(percentile(static_latency, 50), 3),
    }


def main() -> None:
    p = cli.parser(__doc__.splitlines()[0])
    p.add_argument("--out", help="where to save the timeline JSON")
    args = p.parse_args()
    result = run(cli.resolve(args))
    for t in sorted(result["timeline"], key=lambda t: t["sent_s"]):
        print(f"{t['id']:>8}  sent {t['sent_s']:6.3f} s  first token {t['first_token_s']:6.3f} s"
              f"  finished {t['finished_s']:6.3f} s  ({t['tokens']} tokens)")
    cli.show("Continuous batching", {k: v for k, v in result.items() if k != "timeline"})
    out = args.out or (None if args.mock else str(ROOT / "benchmarks" /
                                                  "timeline_continuous_batching.json"))
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
