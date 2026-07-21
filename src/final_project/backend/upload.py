"""Upload prepared chunks to ChromaDB."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import time
from typing import Any

from .chroma_store import get_collection, reset_collection as reset_chroma_collection
from .config import BackendSettings
from .embeddings import EmbeddingClient, get_embedding_client


LOGGER = logging.getLogger(__name__)


def _batched(items: list[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _metadata_value(value: Any) -> str | int | float | bool:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=False)


def chunk_metadata(chunk: dict[str, Any]) -> dict[str, str | int | float | bool]:
    nested_metadata = chunk.get("metadata") or {}
    provenance = chunk.get("provenance") or {}
    metadata = {
        "chunk_id": chunk.get("chunk_id", ""),
        "document_id": chunk.get("document_id", ""),
        "chunk_index": chunk.get("chunk_index", 0),
        "section_type": chunk.get("section_type", ""),
        "heading": chunk.get("heading", ""),
        "section_order": chunk.get("section_order", 0),
        "extraction_method": chunk.get("extraction_method", ""),
        "source_type": chunk.get("source_type", ""),
        "source_identifier": chunk.get("source_identifier", ""),
        "token_count": chunk.get("token_count", 0),
        "sentence_count": chunk.get("sentence_count", 0),
        "title": nested_metadata.get("title", ""),
        "doi": nested_metadata.get("doi", ""),
        "pmid": nested_metadata.get("pmid", ""),
        "pmcid": nested_metadata.get("pmcid", ""),
        "journal": nested_metadata.get("journal", ""),
        "publication_date": nested_metadata.get("publication_date", ""),
        "authors": nested_metadata.get("authors", []),
        "chunker": provenance.get("chunker", ""),
    }
    return {key: _metadata_value(value) for key, value in metadata.items()}


def load_chunks(path: str | Path) -> list[dict[str, Any]]:
    chunks_path = Path(path).expanduser().resolve()
    chunks: list[dict[str, Any]] = []
    with chunks_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            chunk = json.loads(line)
            if not chunk.get("chunk_id"):
                raise ValueError(f"Missing chunk_id at line {line_number}")
            if not chunk.get("text"):
                continue
            chunks.append(chunk)
    return chunks


def _is_quota_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code == 429:
        return True
    message = str(exc).lower()
    return "resource_exhausted" in message or "quota" in message or "rate limit" in message


def _embed_documents_with_quota_retry(
    embedding_client: EmbeddingClient,
    documents: list[str],
    *,
    quota_retry_seconds: float,
    quota_max_retries: int,
) -> list[list[float]]:
    for attempt in range(quota_max_retries + 1):
        try:
            return embedding_client.embed_documents(documents)
        except Exception as exc:
            if not _is_quota_error(exc) or attempt >= quota_max_retries:
                raise
            wait_seconds = max(quota_retry_seconds, 0)
            LOGGER.warning(
                "Embedding quota/rate limit hit; waiting %.1f seconds before retry %s/%s.",
                wait_seconds,
                attempt + 1,
                quota_max_retries,
            )
            time.sleep(wait_seconds)
    raise RuntimeError("Embedding retry loop ended unexpectedly")


def _existing_ids(collection: Any, ids: list[str]) -> set[str]:
    if not ids:
        return set()
    result = collection.get(ids=ids, include=[])
    existing = result.get("ids", []) if isinstance(result, dict) else []
    return {str(item) for item in existing}


def _filter_missing_chunks(
    collection: Any,
    chunks: list[dict[str, Any]],
    *,
    check_batch_size: int,
) -> tuple[list[dict[str, Any]], int]:
    missing_chunks: list[dict[str, Any]] = []
    skipped_existing = 0
    effective_batch_size = max(check_batch_size, 1)
    total_check_batches = math.ceil(len(chunks) / effective_batch_size) if chunks else 0

    LOGGER.info("Checking Chroma for existing chunk IDs before embedding.")
    for batch_index, batch in enumerate(_batched(chunks, effective_batch_size), start=1):
        ids = [chunk["chunk_id"] for chunk in batch]
        existing_ids = _existing_ids(collection, ids)
        if existing_ids:
            skipped_existing += len(existing_ids)
            LOGGER.info(
                "Found %s existing chunks in existence-check batch %s/%s.",
                len(existing_ids),
                batch_index,
                total_check_batches,
            )
        missing_chunks.extend(chunk for chunk in batch if chunk["chunk_id"] not in existing_ids)

    LOGGER.info(
        "Existing-chunk check complete: %s already present, %s still need embedding.",
        skipped_existing,
        len(missing_chunks),
    )
    return missing_chunks, skipped_existing


def upload_chunks(
    chunks_path: str | Path,
    *,
    settings: BackendSettings,
    batch_size: int = 64,
    embedding_client: EmbeddingClient | None = None,
    collection: Any | None = None,
    reset_collection: bool = False,
    skip_existing: bool = False,
    existing_check_batch_size: int = 256,
    batch_sleep_seconds: float = 0,
    quota_retry_seconds: float = 30,
    quota_max_retries: int = 6,
) -> dict[str, Any]:
    chunks = load_chunks(chunks_path)
    LOGGER.info("Loaded %s chunks from %s.", len(chunks), Path(chunks_path).expanduser().resolve())
    LOGGER.info(
        "Preparing embedding client provider=%s model=%s.",
        settings.embedding_provider,
        settings.embedding_model,
    )
    embedding_client = embedding_client or get_embedding_client(settings)
    LOGGER.info("Embedding client ready: %s / %s.", embedding_client.provider, embedding_client.model)
    if reset_collection and collection is not None:
        raise ValueError("reset_collection=True cannot be used with an injected collection")
    if reset_collection:
        LOGGER.warning("Resetting Chroma collection %s before upload.", settings.chroma_collection)
        collection = reset_chroma_collection(settings.chroma_collection)
    else:
        LOGGER.info("Opening Chroma collection %s.", settings.chroma_collection)
        collection = collection or get_collection(create=True)
    total_chunks = len(chunks)
    skipped_existing = 0
    if skip_existing:
        chunks, skipped_existing = _filter_missing_chunks(
            collection,
            chunks,
            check_batch_size=existing_check_batch_size,
        )

    uploaded = 0
    total_batches = math.ceil(len(chunks) / batch_size) if chunks else 0

    for batch_index, batch in enumerate(_batched(chunks, batch_size), start=1):
        ids = [chunk["chunk_id"] for chunk in batch]
        LOGGER.info(
            "Embedding/uploading batch %s/%s (%s chunks, uploaded so far: %s/%s).",
            batch_index,
            total_batches,
            len(batch),
            uploaded,
            len(chunks),
        )
        documents = [chunk["text"] for chunk in batch]
        embeddings = _embed_documents_with_quota_retry(
            embedding_client,
            documents,
            quota_retry_seconds=quota_retry_seconds,
            quota_max_retries=quota_max_retries,
        )
        metadatas = [chunk_metadata(chunk) for chunk in batch]
        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        uploaded += len(batch)
        LOGGER.info("Uploaded batch %s/%s; total uploaded %s/%s.", batch_index, total_batches, uploaded, len(chunks))
        if batch_sleep_seconds > 0 and batch_index < total_batches:
            LOGGER.info("Sleeping %.1f seconds before next embedding batch.", batch_sleep_seconds)
            time.sleep(batch_sleep_seconds)

    collection_count = None
    try:
        collection_count = collection.count()
    except Exception:  # pragma: no cover - optional Chroma method
        LOGGER.debug("Could not read Chroma collection count.", exc_info=True)
    LOGGER.info("Upload complete: %s/%s missing chunks uploaded.", uploaded, len(chunks))
    if skip_existing:
        LOGGER.info("Skipped %s chunks that already existed in Chroma.", skipped_existing)

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "chunks_path": str(Path(chunks_path).expanduser().resolve()),
        "collection": settings.chroma_collection,
        "embedding_provider": embedding_client.provider,
        "embedding_model": embedding_client.model,
        "num_chunks": total_chunks,
        "num_chunks_considered_for_upload": len(chunks),
        "num_uploaded": uploaded,
        "num_skipped_existing": skipped_existing,
        "batch_size": batch_size,
        "reset_collection": reset_collection,
        "skip_existing": skip_existing,
        "existing_check_batch_size": existing_check_batch_size,
        "batch_sleep_seconds": batch_sleep_seconds,
        "quota_retry_seconds": quota_retry_seconds,
        "quota_max_retries": quota_max_retries,
        "collection_count": collection_count,
    }


def check_upload_ready(
    chunks_path: str | Path,
    *,
    settings: BackendSettings,
    embedding_client: EmbeddingClient | None = None,
    collection: Any | None = None,
) -> dict[str, Any]:
    chunks = load_chunks(chunks_path)
    if not chunks:
        raise ValueError(f"No non-empty chunks found in {Path(chunks_path).expanduser().resolve()}")

    LOGGER.info("Loaded %s chunks from %s.", len(chunks), Path(chunks_path).expanduser().resolve())
    LOGGER.info(
        "Preparing embedding client provider=%s model=%s.",
        settings.embedding_provider,
        settings.embedding_model,
    )
    embedding_client = embedding_client or get_embedding_client(settings)
    LOGGER.info("Embedding client ready: %s / %s.", embedding_client.provider, embedding_client.model)

    sample_document = chunks[0]["text"]
    LOGGER.info("Embedding one sample document chunk.")
    document_embedding = embedding_client.embed_documents([sample_document])[0]
    LOGGER.info("Embedding one sample query.")
    query_embedding = embedding_client.embed_query("What does the literature say about dog behaviour?")

    LOGGER.info("Opening Chroma collection %s.", settings.chroma_collection)
    collection = collection or get_collection(create=True)
    collection_count = None
    try:
        collection_count = collection.count()
    except Exception:  # pragma: no cover - optional Chroma method
        LOGGER.debug("Could not read Chroma collection count.", exc_info=True)

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "chunks_path": str(Path(chunks_path).expanduser().resolve()),
        "collection": settings.chroma_collection,
        "embedding_provider": embedding_client.provider,
        "embedding_model": embedding_client.model,
        "num_chunks": len(chunks),
        "sample_document_embedding_dimensions": len(document_embedding),
        "sample_query_embedding_dimensions": len(query_embedding),
        "collection_count": collection_count,
    }
