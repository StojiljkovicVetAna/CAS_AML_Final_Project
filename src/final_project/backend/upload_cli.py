"""CLI for uploading chunk JSONL records to ChromaDB."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from dotenv import load_dotenv

from .config import load_settings
from .upload import check_upload_ready, upload_chunks


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload RAG chunks to ChromaDB")
    parser.add_argument("--chunks", type=Path, default=Path("data/chunks/chunks.jsonl"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--sleep-seconds-between-batches",
        type=float,
        default=0,
        help="Pause after each uploaded batch to stay under embedding API rate limits.",
    )
    parser.add_argument(
        "--quota-retry-seconds",
        type=float,
        default=30,
        help="Seconds to wait before retrying an embedding batch after a quota/rate-limit error.",
    )
    parser.add_argument(
        "--quota-max-retries",
        type=int,
        default=6,
        help="Maximum retries for one embedding batch after quota/rate-limit errors.",
    )
    parser.add_argument("--manifest", type=Path, default=Path("data/chunks/upload_manifest.json"))
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate chunk loading, embedding, and Chroma access without uploading chunks.",
    )
    parser.add_argument(
        "--reset-collection",
        action="store_true",
        help="Delete and recreate the target Chroma collection before uploading.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Check Chroma first and embed/upload only chunk IDs that are missing.",
    )
    parser.add_argument(
        "--existing-check-batch-size",
        type=int,
        default=256,
        help="Number of chunk IDs to check at once when --skip-existing is enabled.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    settings = load_settings()
    if args.check_only:
        if args.reset_collection:
            parser_error = "--reset-collection cannot be combined with --check-only"
            raise SystemExit(parser_error)
        manifest = check_upload_ready(args.chunks, settings=settings)
    else:
        if args.reset_collection and args.skip_existing:
            raise SystemExit("--reset-collection cannot be combined with --skip-existing")
        manifest = upload_chunks(
            args.chunks,
            settings=settings,
            batch_size=args.batch_size,
            reset_collection=args.reset_collection,
            skip_existing=args.skip_existing,
            existing_check_batch_size=args.existing_check_batch_size,
            batch_sleep_seconds=args.sleep_seconds_between_batches,
            quota_retry_seconds=args.quota_retry_seconds,
            quota_max_retries=args.quota_max_retries,
        )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.check_only:
        LOGGER.info(
            "Preflight check succeeded for %s / %s; manifest %s",
            manifest["embedding_provider"],
            manifest["collection"],
            args.manifest,
        )
    else:
        LOGGER.info(
            "Uploaded %s chunks to %s; manifest %s",
            manifest["num_uploaded"],
            manifest["collection"],
            args.manifest,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
