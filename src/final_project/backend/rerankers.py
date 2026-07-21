"""Optional reranking providers."""

from __future__ import annotations

from typing import Any, Protocol

import requests

from .config import BackendSettings


class Reranker(Protocol):
    provider: str

    def rerank(
        self, query: str, chunks: list[dict[str, Any]], *, top_n: int
    ) -> list[dict[str, Any]]:
        ...


class NoReranker:
    provider = "none"

    def rerank(
        self, query: str, chunks: list[dict[str, Any]], *, top_n: int
    ) -> list[dict[str, Any]]:
        return chunks[:top_n]


class JinaReranker:
    provider = "jina"

    def __init__(self, settings: BackendSettings):
        self.api_key = settings.jina_reranker_api_key
        self.url = settings.jina_reranker_url
        self.model = settings.jina_reranker_model
        if not self.api_key:
            raise ValueError("JINA_RERANKER_API_KEY is required when RERANKER_PROVIDER=jina")
        if not self.url:
            raise ValueError("JINA_RERANKER_URL is required when RERANKER_PROVIDER=jina")

    def rerank(
        self, query: str, chunks: list[dict[str, Any]], *, top_n: int
    ) -> list[dict[str, Any]]:
        if not chunks:
            return []
        response = requests.post(
            self.url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "query": query,
                "documents": [chunk["text"] for chunk in chunks],
                "top_n": top_n,
            },
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        reranked: list[dict[str, Any]] = []
        for item in payload.get("results", []):
            original = chunks[int(item["index"])].copy()
            original["rerank_score"] = float(item.get("relevance_score", 0.0))
            reranked.append(original)
        return reranked


def get_reranker(settings: BackendSettings, *, enabled: bool = True) -> Reranker:
    if not enabled:
        return NoReranker()
    provider = settings.reranker_provider
    if provider in {"", "none", "off", "false"}:
        return NoReranker()
    if provider == "jina":
        return JinaReranker(settings)
    raise ValueError(f"Unsupported RERANKER_PROVIDER: {provider}")
