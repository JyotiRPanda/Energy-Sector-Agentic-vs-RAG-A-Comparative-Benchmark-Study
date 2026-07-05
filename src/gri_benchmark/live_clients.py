from __future__ import annotations

import json
import math
import os
import time
import socket
from dataclasses import dataclass
from typing import Any
from time import perf_counter
from urllib import error, request

from gri_benchmark.settings import FoundryConfig, load_foundry_config


@dataclass(frozen=True)
class AzureOpenAIClient:
    endpoint: str
    api_key: str
    model_deployment: str
    embedding_deployment: str
    api_version: str = "2024-06-01"

    def _post_json(self, path: str, payload: dict[str, Any], *, key: str | None = None) -> dict[str, Any]:
        """POST a JSON request with automatic retry on transient network errors."""
        import socket as socket_module
        
        max_retries = 5
        backoff_delays = [0.5, 1, 2, 4, 8]

        last_exc = None
        for attempt in range(max_retries):
            try:
                # Set socket timeout explicitly
                socket_module.setdefaulttimeout(65)
                
                url = f"{self.endpoint}{path}"
                body = json.dumps(payload).encode("utf-8")
                req = request.Request(
                    url,
                    data=body,
                    method="POST",
                    headers={
                        "Content-Type": "application/json",
                        "api-key": key or self.api_key,
                    },
                )
                
                with request.urlopen(req, timeout=65) as resp:
                    resp_data = resp.read()
                    decoded = resp_data.decode("utf-8")
                    return json.loads(decoded)
                
            except error.HTTPError as exc:
                # HTTP errors are permanent, not retryable
                detail = exc.read().decode("utf-8", errors="ignore")
                raise RuntimeError(f"Azure OpenAI HTTP {exc.code}: {detail}") from exc
                
            except Exception as exc:
                last_exc = exc
                exc_str = str(exc)
                exc_name = type(exc).__name__
                
                # Classify as transient or permanent
                is_transient = any([
                    isinstance(exc, (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, socket.timeout, TimeoutError, OSError, socket_module.timeout)),
                    "Connection reset" in exc_str,
                    "Connection aborted" in exc_str,
                    "timeout" in exc_str.lower(),
                    "broken pipe" in exc_str.lower(),
                    "reset by peer" in exc_str.lower(),
                    "connection refused" in exc_str.lower(),
                    "timed out" in exc_str.lower(),
                ])
                
                if not is_transient:
                    # Permanent error
                    print(f"[live-client] Permanent error: {exc_name}: {exc_str[:150]}", flush=True)
                    raise RuntimeError(f"Azure OpenAI permanent error: {exc_name}: {exc_str[:150]}") from exc
                
                # Transient error - retry with backoff
                if attempt < max_retries - 1:
                    wait_time = backoff_delays[attempt]
                    msg = f"[live-client] Transient {exc_name} on attempt {attempt + 1}/{max_retries}. Retrying in {wait_time}s..."
                    print(msg, flush=True)
                    time.sleep(wait_time)
                    continue  # Explicit continue to next retry
                else:
                    # Max retries exhausted
                    msg = f"[live-client] Max retries ({max_retries}) exhausted on {exc_name}"
                    print(msg, flush=True)
                    raise RuntimeError(f"Azure OpenAI failed after {max_retries} retries: {exc_name}: {exc_str[:150]}") from exc
        
        # Should never reach here
        if last_exc:
            raise RuntimeError(f"Unexpected loop exit with exception: {last_exc}") from last_exc
        raise RuntimeError("_post_json: unexpected loop completion")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        result = self.embed_texts_with_usage(texts)
        return result["embeddings"]

    def embed_texts_with_usage(self, texts: list[str]) -> dict[str, Any]:
        start = perf_counter()
        payload = {"input": texts}
        path = f"/openai/deployments/{self.embedding_deployment}/embeddings?api-version={self.api_version}"
        data = self._post_json(path, payload)
        rows = data.get("data", [])
        embeddings = [list(item.get("embedding", [])) for item in rows]
        usage = data.get("usage", {}) if isinstance(data.get("usage", {}), dict) else {}
        return {
            "embeddings": embeddings,
            "usage": {
                "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
                "total_tokens": int(usage.get("total_tokens", 0) or 0),
            },
            "latency_ms": (perf_counter() - start) * 1000,
        }

    def similarity_scores(self, query: str, candidates: list[str]) -> list[float]:
        result = self.similarity_scores_with_usage(query, candidates)
        return result["scores"]

    def similarity_scores_with_usage(self, query: str, candidates: list[str]) -> dict[str, Any]:
        if not candidates:
            return {
                "scores": [],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "latency_ms": 0.0,
            }
        embed = self.embed_texts_with_usage([query] + candidates)
        vectors = embed["embeddings"]
        if len(vectors) != len(candidates) + 1:
            return {
                "scores": [0.0 for _ in candidates],
                "usage": embed["usage"],
                "latency_ms": embed["latency_ms"],
            }

        qv = vectors[0]
        return {
            "scores": [_cosine_similarity(qv, cv) for cv in vectors[1:]],
            "usage": embed["usage"],
            "latency_ms": embed["latency_ms"],
        }

    def generate_grounded_answer(self, question: str, evidence_items: list[dict[str, str]]) -> str:
        result = self.generate_grounded_answer_with_usage(question, evidence_items)
        return str(result["answer"]).strip() or "INSUFFICIENT_CONTEXT"

    def generate_tool_answer(self, *, system_prompt: str, user_prompt: str, max_tokens: int = 96) -> str:
        result = self.generate_tool_answer_with_usage(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
        )
        return str(result["answer"]).strip() or "INSUFFICIENT_CONTEXT"

    def generate_tool_answer_with_usage(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 96,
    ) -> dict[str, Any]:
        start = perf_counter()
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        path = f"/openai/deployments/{self.model_deployment}/chat/completions?api-version={self.api_version}"
        data = self._post_json(path, payload)
        choices = data.get("choices", [])
        if not choices:
            return {
                "answer": "INSUFFICIENT_CONTEXT",
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "latency_ms": (perf_counter() - start) * 1000,
            }
        content = choices[0].get("message", {}).get("content", "")
        text = str(content).strip()
        usage = data.get("usage", {}) if isinstance(data.get("usage", {}), dict) else {}
        return {
            "answer": text or "INSUFFICIENT_CONTEXT",
            "usage": {
                "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
                "total_tokens": int(usage.get("total_tokens", 0) or 0),
            },
            "latency_ms": (perf_counter() - start) * 1000,
        }

    def generate_grounded_answer_with_usage(self, question: str, evidence_items: list[dict[str, str]]) -> dict[str, Any]:
        start = perf_counter()
        evidence_lines = []
        for item in evidence_items:
            evidence_lines.append(
                f"- source_file={item.get('source_file','')} table_id={item.get('table_id','')} value={item.get('value','')} text={item.get('text','')}"
            )

        prompt = (
            "You are answering table-grounded sustainability QA. "
            "Use ONLY the provided evidence lines. "
            "If evidence is insufficient, return exactly INSUFFICIENT_CONTEXT. "
            "Return only the final answer text without explanation.\n\n"
            f"Question: {question}\n"
            "Evidence:\n"
            + "\n".join(evidence_lines)
        )

        payload = {
            "messages": [
                {"role": "system", "content": "Ground answers strictly in provided evidence."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": 64,
        }
        path = f"/openai/deployments/{self.model_deployment}/chat/completions?api-version={self.api_version}"
        data = self._post_json(path, payload)
        choices = data.get("choices", [])
        if not choices:
            return {
                "answer": "INSUFFICIENT_CONTEXT",
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "latency_ms": (perf_counter() - start) * 1000,
            }
        content = choices[0].get("message", {}).get("content", "")
        text = str(content).strip()
        usage = data.get("usage", {}) if isinstance(data.get("usage", {}), dict) else {}
        return {
            "answer": text or "INSUFFICIENT_CONTEXT",
            "usage": {
                "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
                "total_tokens": int(usage.get("total_tokens", 0) or 0),
            },
            "latency_ms": (perf_counter() - start) * 1000,
        }


def estimate_cost_usd(*, prompt_tokens: int, completion_tokens: int, embedding_tokens: int) -> float:
    # Costs are configurable so experiments can use current provider pricing.
    llm_prompt_per_1k = float(os.getenv("LLM_PROMPT_COST_PER_1K", "0.005"))
    llm_completion_per_1k = float(os.getenv("LLM_COMPLETION_COST_PER_1K", "0.015"))
    embedding_per_1k = float(os.getenv("EMBEDDING_COST_PER_1K", "0.0001"))

    return (
        (prompt_tokens / 1000.0) * llm_prompt_per_1k
        + (completion_tokens / 1000.0) * llm_completion_per_1k
        + (embedding_tokens / 1000.0) * embedding_per_1k
    )


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return dot / (na * nb)


def _to_bool(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def maybe_create_live_client(config: FoundryConfig | None = None, *, force: bool = False) -> AzureOpenAIClient | None:
    cfg = config or load_foundry_config()
    use_live = force or _to_bool(os.getenv("USE_LIVE_MODELS"))
    if not use_live:
        return None

    if not cfg.project_endpoint or not cfg.api_key or not cfg.model_deployment or not cfg.embedding_deployment:
        return None

    endpoint = cfg.project_endpoint.rstrip("/")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
    return AzureOpenAIClient(
        endpoint=endpoint,
        api_key=cfg.api_key,
        model_deployment=cfg.model_deployment,
        embedding_deployment=cfg.embedding_deployment,
        api_version=api_version,
    )
