"""05. Prefix caching: a long system prompt shared by every user is read only once.

Most apps send the same long system prompt (rules, catalog, tone) before a short user
question. vLLM keeps the KV-cache blocks of prompts it already read and reuses them when a
new prompt starts with the same tokens, so only the new part pays prefill. This experiment
sends the same questions twice, one at a time: with a shared system prompt, and with a
unique one per request (a different first line breaks the match). It compares time to
first token and reads the prefix-cache hit and query counters from /metrics.

    python experiments/05_prefix_caching.py [--mock]
"""

from __future__ import annotations

import uuid

from vllm_lab import cli, make_client, metrics, stream_chat
from vllm_lab.load import percentile

RULES = (
    "Eres el asistente de soporte de una tienda en línea de Guatemala. Respondes en español, "
    "con amabilidad y en pocas frases. Los envíos dentro de la capital tardan de uno a dos "
    "días hábiles y a los departamentos de tres a cinco. Las devoluciones se aceptan dentro "
    "de treinta días con el comprobante de compra y el producto en su empaque original. "
)
SYSTEM = "Política de atención al cliente, versión 7.\n" + RULES * 30
QUESTIONS = [
    "¿Cuánto tarda un envío a Quetzaltenango?",
    "¿Puedo devolver unos audífonos abiertos?",
    "¿Hacen envíos el fin de semana?",
    "Mi pedido llegó incompleto, ¿qué hago?",
    "¿Puedo cambiar la dirección de entrega?",
    "¿Cuánto cuesta el envío a Petén?",
    "¿Aceptan devoluciones sin caja?",
    "¿Cómo rastreo mi paquete?",
]
MAX_TOKENS = 8


def _counters(base_url: str) -> tuple[float, float]:
    snap = metrics.snapshot(base_url)
    return snap["prefix_cache_hits"] or 0.0, snap["prefix_cache_queries"] or 0.0


def _phase(base_url: str, client, shared: bool) -> dict:
    h0, q0 = _counters(base_url)
    run_id = uuid.uuid4().hex[:8]  # never seen before, even if you rerun against the same server
    ttft, prompt_tokens = [], 0
    for i, question in enumerate(QUESTIONS):
        system = SYSTEM if shared else f"Sesión única {run_id}-{i}.\n" + SYSTEM
        t = stream_chat(client, question, max_tokens=MAX_TOKENS, system=system)
        ttft.append(t.ttft_s)
        prompt_tokens = t.prompt_tokens
    h1, q1 = _counters(base_url)
    hits, queries = h1 - h0, q1 - q0
    return {
        "requests": len(QUESTIONS),
        "prompt_tokens": prompt_tokens,
        "p50_ttft_ms": round(percentile(ttft, 50) * 1000, 1),
        "p95_ttft_ms": round(percentile(ttft, 95) * 1000, 1),
        "prefix_cache_hits": int(hits),
        "prefix_cache_queries": int(queries),
        "hit_rate_pct": round(100 * hits / queries, 1) if queries else None,
    }


def run(base_url: str) -> dict:
    client = make_client(base_url)
    stream_chat(client, "Hola", max_tokens=8)  # warmup
    stream_chat(client, "Hola", max_tokens=1, system=SYSTEM)  # the first reader pays in full
    shared = _phase(base_url, client, shared=True)
    unique = _phase(base_url, client, shared=False)
    return {
        "shared_prefix": shared,
        "unique_prefix": unique,
        "ttft_speedup_x": round(unique["p50_ttft_ms"] / max(shared["p50_ttft_ms"], 1e-9), 2),
        "counters_found": metrics.pick(metrics.fetch(base_url), metrics.PREFIX_QUERIES)
        is not None,
    }


def main() -> None:
    args = cli.parse(__doc__.splitlines()[0])
    cli.show("Prefix caching", run(cli.resolve(args)))


if __name__ == "__main__":
    main()
