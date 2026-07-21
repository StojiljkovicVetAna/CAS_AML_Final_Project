"""Command-line interface for creating RAG chunks from extracted documents."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .models import ChunkConfig
from .pipeline import chunk_document_file, chunk_document_dir, write_chunks_jsonl


LOGGER = logging.getLogger(__name__)


def _config_from_args(args: argparse.Namespace) -> ChunkConfig:
    return ChunkConfig(
        target_tokens=args.target_tokens,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap_tokens,
        min_tokens=args.min_tokens,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunk extracted paper JSON for RAG")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--target-tokens", type=int, default=400)
    parser.add_argument("--max-tokens", type=int, default=520)
    parser.add_argument("--overlap-tokens", type=int, default=80)
    parser.add_argument("--min-tokens", type=int, default=80)
    commands = parser.add_subparsers(dest="command", required=True)

    document = commands.add_parser("document", help="Chunk one normalized document JSON")
    document.add_argument("path", type=Path)
    document.add_argument("--output", type=Path, default=Path("data/chunks/chunks.jsonl"))

    corpus = commands.add_parser("corpus", help="Chunk all document JSON files")
    corpus.add_argument(
        "--document-dir",
        type=Path,
        default=Path("data/extracted/documents"),
    )
    corpus.add_argument("--output-dir", type=Path, default=Path("data/chunks"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    config = _config_from_args(args)

    if args.command == "document":
        chunks = chunk_document_file(args.path, config=config)
        output_path = write_chunks_jsonl(chunks, args.output)
        LOGGER.info("Wrote %s chunks to %s", len(chunks), output_path)
        return 0

    manifest = chunk_document_dir(
        args.document_dir,
        output_dir=args.output_dir,
        config=config,
    )
    LOGGER.info(
        "Chunked %s documents into %s chunks; manifest %s",
        manifest["num_documents"],
        manifest["num_chunks"],
        manifest["manifest_path"],
    )
    if manifest["failures"]:
        LOGGER.error(json.dumps(manifest["failures"], ensure_ascii=False, indent=2))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
