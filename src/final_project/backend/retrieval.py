"""Vector retrieval and context formatting."""

from __future__ import annotations

import json
import time
from typing import Any

from .chroma_store import get_collection
from .config import BackendSettings
from .deployment import DISABLED_RERANKER_PROVIDERS, pipeline_uses_reranker
from .embeddings import EmbeddingClient, get_embedding_client
from .rerankers import get_reranker


def _parse_authors(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except json.JSONDecodeError:
            return [part.strip() for part in value.split(";") if part.strip()]
    return []


def _format_chunk(
    *,
    document: str,
    metadata: dict[str, Any],
    distance: float | None,
) -> dict[str, Any]:
    return {
        "text": document,
        "chunk_id": str(metadata.get("chunk_id", "")),
        "document_id": str(metadata.get("document_id", "")),
        "title": str(metadata.get("title", "")),
        "authors": _parse_authors(metadata.get("authors", "")),
        "doi": str(metadata.get("doi", "")),
        "journal": str(metadata.get("journal", "")),
        "publication_date": str(metadata.get("publication_date", "")),
        "section_type": str(metadata.get("section_type", "")),
        "heading": str(metadata.get("heading", "")),
        "chunk_index": int(metadata.get("chunk_index", 0) or 0),
        "distance": distance,
        "rerank_score": None,
    }


def retrieve_chunks(
    query: str,
    *,
    settings: BackendSettings,
    top_k: int | None = None,
    top_n: int | None = None,
    use_reranker: bool | None = None,
    embedding_client: EmbeddingClient | None = None,
    collection: Any | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    timings_ms: dict[str, int] = {}
    effective_top_k = top_k or settings.retrieve_top_k
    effective_top_n = top_n or settings.context_top_n
    reranker_enabled = (
        pipeline_uses_reranker(settings.rag_pipeline)
        and settings.reranker_provider not in DISABLED_RERANKER_PROVIDERS
        if use_reranker is None
        else use_reranker
    )

    t0 = time.perf_counter()
    embedding_client = embedding_client or get_embedding_client(settings)
    query_embedding = embedding_client.embed_query(query)
    timings_ms["embed_query"] = int((time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    collection = collection or get_collection(create=False)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=effective_top_k,
        include=["documents", "metadatas", "distances"],
    )
    timings_ms["chroma_query"] = int((time.perf_counter() - t0) * 1000)

    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]
    chunks = [
        _format_chunk(document=document, metadata=metadata or {}, distance=distance)
        for document, metadata, distance in zip(documents, metadatas, distances)
    ]

    t0 = time.perf_counter()
    reranker = get_reranker(settings, enabled=reranker_enabled)
    chunks = reranker.rerank(query, chunks, top_n=effective_top_n)
    timings_ms["rerank"] = int((time.perf_counter() - t0) * 1000)
    return chunks, timings_ms


def format_context_for_llm(chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return "No relevant context found."
    parts = []
    for index, chunk in enumerate(chunks, start=1):
        title = chunk.get("title") or chunk.get("document_id") or "Unknown source"
        heading = chunk.get("heading") or chunk.get("section_type") or "section"
        doi = chunk.get("doi")
        source_line = f"[Source {index}] {title}; {heading}"
        if doi:
            source_line += f"; DOI: {doi}"
        parts.append(f"{source_line}\n{chunk['text']}")
    return "\n\n".join(parts)
