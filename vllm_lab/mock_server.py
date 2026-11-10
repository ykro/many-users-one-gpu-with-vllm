"""A tiny fake of vLLM's OpenAI-compatible server that also fakes its scheduler.

Same HTTP as `vllm serve` (/health, /v1/models, /v1/chat/completions with and without
streaming, /metrics in Prometheus text), plus a toy engine that behaves like vLLM's:

- Continuous batching. One engine loop runs steps. In every step each running request
  gets one new token, and new requests join (or finished ones leave) between two steps, so
  a short answer never waits for a long one. A step costs a fixed part plus a little per
  running request: more users means more tokens per step, each user a bit slower.
- Paged KV-cache. Memory is a pool of blocks of BLOCK_SIZE tokens. A request takes
  blocks as its context grows, never a contiguous reservation for its maximum length. When
  the pool is full, new requests wait, and if a running request needs a block that does not
  exist, the newest running request is preempted (its blocks are freed, it goes back to
  the front of the queue and its context is recomputed later).
- Prefix caching. Full prompt blocks are hashed as a chain. A new prompt that starts with
  blocks already seen only pays prefill for the rest.

Its numbers are made up on purpose; only the live judge (`make verify-live`) measures a
real GPU.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import queue
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STEP_BASE_S = 0.003        # one engine step with a single request (decode of one token)
STEP_PER_SEQ_S = 0.0003    # each extra running request makes the step 10 % longer
PREFILL_PER_TOKEN_S = 0.00002
BLOCK_SIZE = 16            # tokens per KV block, as in vLLM
NUM_BLOCKS = 256           # 4096 tokens of KV-cache in total: small on purpose
MAX_NUM_SEQS = 64
PREFIX_CACHE_BLOCKS = 4096
NATURAL_LENGTH = 40
CHARS_PER_TOKEN = 4

_SENTENCE = (
    "Un servidor de inferencia atiende a muchos usuarios con una sola GPU. En cada paso "
    "agrega un token a cada respuesta activa, y las peticiones nuevas entran al batch "
    "en cuanto hay memoria libre."
)
WORDS = _SENTENCE.split(" ")


def split_tokens(text: str) -> list[str]:
    """Fake tokenizer: one token every CHARS_PER_TOKEN characters."""
    return [text[i : i + CHARS_PER_TOKEN] for i in range(0, len(text), CHARS_PER_TOKEN)] or [""]


def block_hashes(tokens: list[str]) -> list[str]:
    """Hash of every full block, chained, so a block only matches with the same past."""
    hashes, prev = [], ""
    for start in range(0, len(tokens) - BLOCK_SIZE + 1, BLOCK_SIZE):
        prev = hashlib.sha256((prev + "".join(tokens[start : start + BLOCK_SIZE])).encode()
                              ).hexdigest()
        hashes.append(prev)
    return hashes


def blocks_for(tokens: int) -> int:
    return math.ceil(tokens / BLOCK_SIZE)


def fake_text(prompt: str, n: int) -> list[str]:
    start = int(hashlib.sha256(prompt.encode()).hexdigest(), 16) % len(WORDS)
    return [WORDS[(start + i) % len(WORDS)] + " " for i in range(n)]


@dataclass
class Seq:
    seq_id: int
    prompt_tokens: int
    target: int
    hashes: list[str]
    out: queue.Queue = field(default_factory=queue.Queue)
    generated: int = 0
    blocks: int = 0
    prefilled: bool = False

    @property
    def context(self) -> int:
        return self.prompt_tokens + self.generated


class Engine:
    """The scheduler loop. Handlers submit sequences; the loop feeds tokens to their queues."""

    def __init__(self, num_blocks: int = NUM_BLOCKS):
        self.num_blocks = num_blocks
        self.free_blocks = num_blocks
        self.waiting: deque[Seq] = deque()
        self.running: list[Seq] = []
        self.prefix_cache: OrderedDict[str, None] = OrderedDict()
        self.counters = {"preemptions": 0, "prefix_hits": 0, "prefix_queries": 0,
                         "prompt_tokens": 0, "generation_tokens": 0}
        self._ids = itertools.count()
        self._cv = threading.Condition()
        threading.Thread(target=self._loop, daemon=True).start()

    # --- called from HTTP handler threads -------------------------------------------------
    def submit(self, prompt: str, target: int) -> tuple[Seq, int]:
        tokens = split_tokens(prompt)
        seq = Seq(next(self._ids), len(tokens), target, block_hashes(tokens))
        with self._cv:
            self.waiting.append(seq)
            self._cv.notify()
        return seq, len(tokens)

    def metrics_text(self) -> str:
        with self._cv:
            c = dict(self.counters)
            used = 1 - self.free_blocks / self.num_blocks
            running, waiting = len(self.running), len(self.waiting)
        lbl = '{model_name="gemma"}'
        cfg = (f'{{block_size="{BLOCK_SIZE}",num_gpu_blocks="{self.num_blocks}",'
               f'enable_prefix_caching="True"}}')
        lines = [
            ("vllm:num_requests_running", "gauge", lbl, running),
            ("vllm:num_requests_waiting", "gauge", lbl, waiting),
            ("vllm:kv_cache_usage_perc", "gauge", lbl, used),
            ("vllm:num_preemptions_total", "counter", lbl, c["preemptions"]),
            ("vllm:prefix_cache_queries_total", "counter", lbl, c["prefix_queries"]),
            ("vllm:prefix_cache_hits_total", "counter", lbl, c["prefix_hits"]),
            ("vllm:prompt_tokens_total", "counter", lbl, c["prompt_tokens"]),
            ("vllm:generation_tokens_total", "counter", lbl, c["generation_tokens"]),
            ("vllm:cache_config_info", "gauge", cfg, 1),
        ]
        out = []
        for name, kind, labels, value in lines:
            out += [f"# TYPE {name} {kind}", f"{name}{labels} {float(value)}"]
        return "\n".join(out) + "\n"

    # --- the engine loop ------------------------------------------------------------------
    def _loop(self) -> None:
        while True:
            with self._cv:
                while not self.waiting and not self.running:
                    self._cv.wait()
                prefill_tokens = self._schedule()
                batch = list(self.running)
            time.sleep(STEP_BASE_S + STEP_PER_SEQ_S * (len(batch) - 1)
                       + PREFILL_PER_TOKEN_S * prefill_tokens)
            with self._cv:
                for seq in batch:
                    seq.generated += 1
                    self.counters["generation_tokens"] += 1
                    seq.out.put(seq.generated - 1)
                    if seq.generated >= seq.target:
                        self.running.remove(seq)
                        self.free_blocks += seq.blocks
                        seq.out.put(None)

    def _schedule(self) -> int:
        """Grows running requests by one token, preempting if needed, then admits new ones."""
        for seq in list(self.running):
            if seq not in self.running:
                continue
            need = blocks_for(seq.context + 1) - seq.blocks
            while need > self.free_blocks and self.running:
                self._preempt(self.running[-1])  # newest first, like vLLM
                if seq not in self.running:
                    break
            if seq in self.running and need > 0:
                self.free_blocks -= need
                seq.blocks += need
        prefill = 0
        while self.waiting and len(self.running) < MAX_NUM_SEQS:
            seq = self.waiting[0]
            need = blocks_for(seq.context + 1)
            if need > self.free_blocks and self.running:
                break  # memory is full: it waits, which is what /metrics shows
            self.waiting.popleft()
            take = min(need, self.free_blocks)
            self.free_blocks -= take
            seq.blocks = take
            self.running.append(seq)
            prefill += self._prefill_cost(seq)
        return prefill

    def _preempt(self, seq: Seq) -> None:
        self.running.remove(seq)
        self.free_blocks += seq.blocks
        seq.blocks = 0
        self.waiting.appendleft(seq)
        self.counters["preemptions"] += 1

    def _prefill_cost(self, seq: Seq) -> int:
        cached = 0
        for h in seq.hashes:
            if h not in self.prefix_cache:
                break
            self.prefix_cache.move_to_end(h)
            cached += BLOCK_SIZE
        for h in seq.hashes:
            self.prefix_cache[h] = None
        while len(self.prefix_cache) > PREFIX_CACHE_BLOCKS:
            self.prefix_cache.popitem(last=False)
        self.counters["prefix_queries"] += seq.prompt_tokens
        self.counters["prefix_hits"] += cached
        if not seq.prefilled:
            self.counters["prompt_tokens"] += seq.prompt_tokens
        seq.prefilled = True
        # After a preemption the generated tokens are recomputed too.
        return seq.prompt_tokens - cached + seq.generated


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # keep the terminal clean
        pass

    def _send(self, code: int, data: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, code: int, body: dict) -> None:
        self._send(code, json.dumps(body).encode(), "application/json")

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {})
        elif self.path == "/metrics":
            self._send(200, self.server.engine.metrics_text().encode(),
                       "text/plain; version=0.0.4")
        elif self.path == "/v1/models":
            self._json(200, {"object": "list", "data": [
                {"id": "gemma", "object": "model", "owned_by": "vllm", "max_model_len": 8192,
                 "root": "google/gemma-4-E4B-it"}]})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self._json(404, {"error": "not found"})
            return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = "".join(m["content"] for m in body["messages"])
        max_tokens = int(body.get("max_tokens") or 256)
        n = max_tokens if body.get("ignore_eos") else min(max_tokens, NATURAL_LENGTH)
        finish = "length" if n == max_tokens else "stop"
        seq, ptoks = self.server.engine.submit(prompt, n)
        words = fake_text(prompt, n)
        usage = {"prompt_tokens": ptoks, "completion_tokens": n, "total_tokens": ptoks + n}
        if not body.get("stream"):
            while seq.out.get() is not None:
                pass
            self._json(200, {
                "id": "chatcmpl-mock", "object": "chat.completion", "model": "gemma",
                "choices": [{"index": 0, "finish_reason": finish,
                             "message": {"role": "assistant", "content": "".join(words)}}],
                "usage": usage})
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            while (i := seq.out.get()) is not None:
                self._chunk({"id": "chatcmpl-mock", "object": "chat.completion.chunk",
                             "model": "gemma", "choices": [{"index": 0,
                                                            "delta": {"content": words[i]},
                                                            "finish_reason": None}]})
            self._chunk({"id": "chatcmpl-mock", "object": "chat.completion.chunk",
                         "model": "gemma",
                         "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
            if (body.get("stream_options") or {}).get("include_usage"):
                self._chunk({"id": "chatcmpl-mock", "object": "chat.completion.chunk",
                             "model": "gemma", "choices": [], "usage": usage})
            self._raw(b"data: [DONE]\n\n")
            self._raw(b"")
        except OSError:  # client went away; the engine finishes the sequence on its own
            pass

    def _chunk(self, obj: dict) -> None:
        self._raw(f"data: {json.dumps(obj)}\n\n".encode())

    def _raw(self, data: bytes) -> None:
        self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()


class _QuietServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 256  # 64 users connect at once; the default backlog of 5 is too small

    def handle_error(self, request, client_address):  # clients closing keep-alive sockets
        pass


def start(port: int = 0, num_blocks: int = NUM_BLOCKS) -> tuple[ThreadingHTTPServer, str]:
    """Starts the mock in a background thread; returns the server and its /v1 base URL."""
    server = _QuietServer(("127.0.0.1", port), Handler)
    server.engine = Engine(num_blocks)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/v1"


if __name__ == "__main__":
    srv, url = start(8000)
    print(f"Mock vLLM server at {url} (Ctrl+C to stop)")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        srv.shutdown()
