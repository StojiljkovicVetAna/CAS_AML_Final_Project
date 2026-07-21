"""Section-aware chunking for normalized extraction JSON files."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from final_project.extraction.io import safe_filename

from .models import ChunkConfig, TextChunk
from .text import estimate_token_count, is_caption_or_notice, split_paragraphs, split_sentences


def _heading_is_excluded(heading: str, config: ChunkConfig) -> bool:
    normalized = heading.strip().lower()
    return bool(normalized) and any(
        excluded in normalized for excluded in config.exclude_headings
    )


def _section_sentences(section: dict[str, Any]) -> list[str]:
    sentences: list[str] = []
    for paragraph in split_paragraphs(str(section.get("text", ""))):
        if is_caption_or_notice(paragraph):
            continue
        sentences.extend(split_sentences(paragraph))
    return sentences


def _split_oversized_sentence(sentence: str, config: ChunkConfig) -> list[str]:
    if estimate_token_count(sentence) <= config.max_tokens:
        return [sentence]

    pieces: list[str] = []
    current_words: list[str] = []
    for word in sentence.split():
        candidate = " ".join([*current_words, word])
        if current_words and estimate_token_count(candidate) > config.target_tokens:
            pieces.append(" ".join(current_words).strip())
            current_words = [word]
        else:
            current_words.append(word)

    if current_words:
        pieces.append(" ".join(current_words).strip())
    return [piece for piece in pieces if piece]


def _chunk_sentence_windows(
    sentences: list[str],
    *,
    config: ChunkConfig,
) -> list[list[str]]:
    if not sentences:
        return []

    sentence_tokens = [estimate_token_count(sentence) for sentence in sentences]
    windows: list[list[str]] = []
    start = 0

    while start < len(sentences):
        end = start
        token_count = 0

        while end < len(sentences):
            next_count = sentence_tokens[end]
            would_exceed_target = (
                end > start and token_count + next_count > config.target_tokens
            )
            would_exceed_max = (
                end > start and token_count + next_count > config.max_tokens
            )
            if would_exceed_target or would_exceed_max:
                break
            token_count += next_count
            end += 1

        if end == start:
            end += 1

        windows.append(sentences[start:end])
        if end >= len(sentences):
            break

        overlap_start = end
        overlap_count = 0
        while overlap_start > start and overlap_count < config.overlap_tokens:
            overlap_start -= 1
            overlap_count += sentence_tokens[overlap_start]

        start = overlap_start if overlap_start > start else end

    return windows


def chunk_document(
    document: dict[str, Any],
    *,
    config: ChunkConfig | None = None,
) -> list[TextChunk]:
    """Create RAG-ready chunks from one normalized extraction document."""
    config = config or ChunkConfig()
    document_id = str(document.get("document_id", "document"))
    document_metadata = document.get("metadata", {}) or {}
    source_type = str(document.get("source_type", ""))
    source_identifier = str(document.get("source_identifier", ""))
    extraction_method = str(document.get("extraction_method", ""))
    chunk_base = safe_filename(document_id)
    chunks: list[TextChunk] = []

    for section in document.get("sections", []) or []:
        section_type = str(section.get("section_type", ""))
        heading = str(section.get("heading", ""))
        if section_type not in config.include_section_types:
            continue
        if _heading_is_excluded(heading, config):
            continue

        sentences = []
        for sentence in _section_sentences(section):
            sentences.extend(_split_oversized_sentence(sentence, config))
        for sentence_window in _chunk_sentence_windows(sentences, config=config):
            text = " ".join(sentence_window).strip()
            if not text:
                continue
            token_count = estimate_token_count(text)
            if token_count < config.min_tokens and chunks:
                previous = chunks[-1]
                if (
                    previous.document_id == document_id
                    and previous.section_order == int(section.get("order", 0))
                    and previous.token_count + token_count <= config.max_tokens
                ):
                    previous.text = f"{previous.text} {text}".strip()
                    previous.token_count = estimate_token_count(previous.text)
                    previous.sentence_count += len(sentence_window)
                    continue

            chunk_index = len(chunks)
            chunk_id = f"{chunk_base}_chunk_{chunk_index:05d}"
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    text=text,
                    chunk_index=chunk_index,
                    section_type=section_type,
                    heading=heading,
                    section_order=int(section.get("order", 0)),
                    extraction_method=extraction_method,
                    source_type=source_type,
                    source_identifier=source_identifier,
                    token_count=token_count,
                    sentence_count=len(sentence_window),
                    metadata={
                        "title": document_metadata.get("title", ""),
                        "doi": document_metadata.get("doi", ""),
                        "pmid": document_metadata.get("pmid", ""),
                        "pmcid": document_metadata.get("pmcid", ""),
                        "journal": document_metadata.get("journal", ""),
                        "publication_date": document_metadata.get("publication_date", ""),
                        "authors": document_metadata.get("authors", []),
                    },
                    provenance={
                        "chunker": "section_aware_academic_v1",
                        "target_tokens": config.target_tokens,
                        "max_tokens": config.max_tokens,
                        "overlap_tokens": config.overlap_tokens,
                        "source_schema_version": document.get("schema_version", ""),
                    },
                )
            )

    return chunks


def chunk_document_file(
    path: str | Path,
    *,
    config: ChunkConfig | None = None,
) -> list[TextChunk]:
    source_path = Path(path).expanduser().resolve()
    document = json.loads(source_path.read_text(encoding="utf-8"))
    return chunk_document(document, config=config)


def write_chunks_jsonl(chunks: list[TextChunk], path: str | Path) -> Path:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
    return output_path


def chunk_document_dir(
    document_dir: str | Path,
    *,
    output_dir: str | Path,
    config: ChunkConfig | None = None,
) -> dict[str, Any]:
    """Chunk every normalized document JSON in a directory and write JSONL output."""
    config = config or ChunkConfig()
    document_path = Path(document_dir).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    all_chunks: list[TextChunk] = []
    processed: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for path in sorted(document_path.glob("*.json")):
        try:
            chunks = chunk_document_file(path, config=config)
            all_chunks.extend(chunks)
            processed.append(
                {
                    "document_path": str(path),
                    "document_id": chunks[0].document_id if chunks else path.stem,
                    "num_chunks": len(chunks),
                    "total_tokens": sum(chunk.token_count for chunk in chunks),
                }
            )
        except Exception as exc:  # pragma: no cover - preserved in manifest
            failures.append({"document_path": str(path), "error": str(exc)})

    chunks_path = write_chunks_jsonl(all_chunks, output_path / "chunks.jsonl")
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "chunk_schema_version": "1.0",
        "chunker": "section_aware_academic_v1",
        "config": {
            "target_tokens": config.target_tokens,
            "max_tokens": config.max_tokens,
            "overlap_tokens": config.overlap_tokens,
            "min_tokens": config.min_tokens,
            "include_section_types": list(config.include_section_types),
            "exclude_headings": list(config.exclude_headings),
        },
        "document_dir": str(document_path),
        "chunks_path": str(chunks_path),
        "num_documents": len(processed),
        "num_chunks": len(all_chunks),
        "processed": processed,
        "failures": failures,
    }
    manifest_path = output_path / "chunk_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest["manifest_path"] = str(manifest_path)
    return manifest
