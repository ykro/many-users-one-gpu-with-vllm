"""OpenAI-compatible clients (sync and async) pointed at your own vLLM server.

The only thing that changes from a hosted API is BASE_URL. With the Cloud Run tunnel
(`gcloud run services proxy vllm-gemma --region us-east4 --port 8000`) the server is
at http://localhost:8000/v1 and no API key is needed: the tunnel signs each request with
your gcloud identity.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

from openai import AsyncOpenAI, OpenAI

BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
MODEL = os.environ.get("VLLM_MODEL_NAME", "gemma")


def make_client(base_url: str | None = None) -> OpenAI:
    # vLLM ignores the key unless it was started with --api-key; the SDK just needs a string.
    return OpenAI(base_url=base_url or BASE_URL, api_key=os.environ.get("VLLM_API_KEY", "local"),
                  timeout=900, max_retries=0)


def make_async_client(base_url: str | None = None) -> AsyncOpenAI:
    # max_retries=0: a retried request would hide exactly the queueing we want to measure.
    return AsyncOpenAI(base_url=base_url or BASE_URL,
                       api_key=os.environ.get("VLLM_API_KEY", "local"), timeout=900,
                       max_retries=0)


def messages(prompt: str, system: str | None = None) -> list[dict]:
    msgs = [{"role": "system", "content": system}] if system else []
    return msgs + [{"role": "user", "content": prompt}]


@dataclass
class StreamTiming:
    """One streamed answer: when it was sent, when the first token arrived, when it ended."""

    sent_at: float = 0.0  # time.perf_counter() when the request left the client
    ttft_s: float = 0.0
    total_s: float = 0.0
    chunks: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    text: str = ""
    arrivals: list[float] = field(default_factory=list)

    @property
    def decode_s(self) -> float:
        return max(self.total_s - self.ttft_s, 1e-9)

    @property
    def itl_s(self) -> float:
        """Mean inter-token latency: the gap between two tokens once the answer started."""
        return self.decode_s / max(self.completion_tokens - 1, 1)

    @property
    def decode_tokens_per_s(self) -> float:
        return max(self.completion_tokens - 1, 0) / self.decode_s


def _request_args(prompt: str, system: str | None, max_tokens: int, ignore_eos: bool) -> dict:
    return {
        "model": MODEL,
        "messages": messages(prompt, system),
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 7,
        "stream": True,
        "stream_options": {"include_usage": True},
        "extra_body": {"ignore_eos": True} if ignore_eos else None,
    }


def _consume(timing: StreamTiming, chunk, now: float, parts: list[str]) -> None:
    if chunk.usage is not None:
        timing.prompt_tokens = chunk.usage.prompt_tokens
        timing.completion_tokens = chunk.usage.completion_tokens
    if chunk.choices and chunk.choices[0].delta.content:
        if timing.chunks == 0:
            timing.ttft_s = now
        timing.chunks += 1
        timing.arrivals.append(now)
        parts.append(chunk.choices[0].delta.content)


def stream_chat(
    client: OpenAI,
    prompt: str,
    max_tokens: int = 256,
    system: str | None = None,
    ignore_eos: bool = False,
) -> StreamTiming:
    """Streams one answer (blocking) and records when each chunk arrives."""
    timing = StreamTiming(sent_at=time.perf_counter())
    stream = client.chat.completions.create(**_request_args(prompt, system, max_tokens,
                                                            ignore_eos))
    parts: list[str] = []
    for chunk in stream:
        _consume(timing, chunk, time.perf_counter() - timing.sent_at, parts)
    timing.total_s = time.perf_counter() - timing.sent_at
    timing.text = "".join(parts)
    return timing


async def astream_chat(
    client: AsyncOpenAI,
    prompt: str,
    max_tokens: int = 256,
    system: str | None = None,
    ignore_eos: bool = False,
) -> StreamTiming:
    """Same as stream_chat, but many of these can be in flight at the same time."""
    timing = StreamTiming(sent_at=time.perf_counter())
    stream = await client.chat.completions.create(**_request_args(prompt, system, max_tokens,
                                                                  ignore_eos))
    parts: list[str] = []
    async for chunk in stream:
        _consume(timing, chunk, time.perf_counter() - timing.sent_at, parts)
    timing.total_s = time.perf_counter() - timing.sent_at
    timing.text = "".join(parts)
    return timing
