"""Command-line interface for extraction module one."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from .corpus import run_corpus
from .errors import ExtractionError
from .grobid import GrobidExtractor
from .io import write_document, write_manifest
from .pipeline import ExtractionPipeline


LOGGER = logging.getLogger(__name__)


def _common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/extracted")
    )
    parser.add_argument(
        "--grobid-url",
        default=os.getenv("GROBID_URL", "http://127.0.0.1:8070"),
    )
    parser.add_argument("--save-raw-xml", action="store_true")
    return parser


def parse_args() -> argparse.Namespace:
    common = _common_parser()
    parser = argparse.ArgumentParser(description="Extract scientific papers for RAG")
    parser.add_argument("--verbose", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)

    pdf = commands.add_parser("pdf", parents=[common], help="Extract one local PDF")
    pdf.add_argument("path", type=Path)
    pdf.add_argument("--no-fallback", action="store_true")

    pdf_dir = commands.add_parser(
        "pdf-dir", parents=[common], help="Extract every PDF in a directory"
    )
    pdf_dir.add_argument("path", type=Path)
    pdf_dir.add_argument("--recursive", action="store_true")
    pdf_dir.add_argument("--no-fallback", action="store_true")

    online = commands.add_parser(
        "online", parents=[common], help="Fetch PMC full text by identifier"
    )
    identifier = online.add_mutually_exclusive_group(required=True)
    identifier.add_argument("--pmcid")
    identifier.add_argument("--pmid")
    identifier.add_argument("--doi")
    online.add_argument("--email", default=os.getenv("NCBI_EMAIL"))
    online.add_argument("--ncbi-api-key", default=os.getenv("NCBI_API_KEY"))
    online.add_argument(
        "--allow-abstract-fallback",
        action="store_true",
        help="Allow an explicitly marked abstract-only result when PMC full text is unavailable",
    )

    corpus = commands.add_parser(
        "corpus", parents=[common], help="Incrementally extract PDFs and DOI-only papers"
    )
    corpus.add_argument(
        "--input-dir", type=Path, default=Path("input_literature")
    )
    corpus.add_argument("--pdf-dir", type=Path)
    corpus.add_argument("--doi-list", type=Path)
    corpus.add_argument("--recursive", action="store_true")
    corpus.add_argument("--no-fallback", action="store_true")
    return parser.parse_args()


def _pipeline(args: argparse.Namespace) -> ExtractionPipeline:
    return ExtractionPipeline(
        grobid_extractor=GrobidExtractor(args.grobid_url),
        enable_pdf_fallback=not getattr(args, "no_fallback", False),
    )


def _processed_entry(document, path: Path) -> dict[str, object]:
    return {
        "document_id": document.document_id,
        "output_path": str(path),
        "extraction_method": document.extraction_method,
        "fallback_used": bool(document.provenance.get("fallback_used")),
        "section_count": len(document.sections),
        "character_count": document.diagnostics.get("character_count", 0),
    }


def main() -> int:
    load_dotenv()
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    pipeline = _pipeline(args)

    try:
        if args.command == "corpus":
            pdf_dir = args.pdf_dir or args.input_dir / "PDFs"
            doi_list = args.doi_list or args.input_dir / "list_DOIs.txt"
            result = run_corpus(
                pipeline,
                pdf_dir=pdf_dir,
                doi_list=doi_list,
                output_dir=args.output_dir,
                recursive=args.recursive,
                save_raw_xml=args.save_raw_xml,
            )
            for item in result.processed:
                LOGGER.info("Extracted %s using %s", item["source"], item["extraction_method"])
            for item in result.skipped:
                LOGGER.info("Skipped %s: %s", item["source"], item["reason"])
            for item in result.failures:
                LOGGER.error("Failed %s: %s", item["source"], item["error"])
            LOGGER.info(
                "Corpus run: %s processed, %s skipped, %s failed; manifest %s",
                len(result.processed),
                len(result.skipped),
                len(result.failures),
                result.manifest_path,
            )
            return 1 if result.failures else 0

        if args.command == "pdf":
            document = pipeline.extract_pdf(args.path)
            path = write_document(
                document, args.output_dir, save_raw_xml=args.save_raw_xml
            )
            write_manifest(
                args.output_dir,
                command="pdf",
                processed=[_processed_entry(document, path)],
                failures=[],
            )
            LOGGER.info("Wrote %s using %s", path, document.extraction_method)
            return 0

        if args.command == "online":
            document = pipeline.extract_online(
                pmcid=args.pmcid,
                pmid=args.pmid,
                doi=args.doi,
                allow_abstract_fallback=args.allow_abstract_fallback,
                email=args.email,
                ncbi_api_key=args.ncbi_api_key,
            )
            path = write_document(
                document, args.output_dir, save_raw_xml=args.save_raw_xml
            )
            write_manifest(
                args.output_dir,
                command="online",
                processed=[_processed_entry(document, path)],
                failures=[],
            )
            LOGGER.info("Wrote %s using %s", path, document.extraction_method)
            return 0

        pattern = "**/*.pdf" if args.recursive else "*.pdf"
        pdf_paths = sorted(args.path.expanduser().resolve().glob(pattern))
        processed = []
        failures = []
        for pdf_path in pdf_paths:
            try:
                document = pipeline.extract_pdf(pdf_path)
                path = write_document(
                    document, args.output_dir, save_raw_xml=args.save_raw_xml
                )
                processed.append(_processed_entry(document, path))
                LOGGER.info("Extracted %s using %s", pdf_path.name, document.extraction_method)
            except ExtractionError as exc:
                LOGGER.error("Failed %s: %s", pdf_path.name, exc)
                failures.append({"source": str(pdf_path), "error": str(exc)})
        write_manifest(
            args.output_dir,
            command="pdf-dir",
            processed=processed,
            failures=failures,
        )
        LOGGER.info("Processed %s PDFs; %s failures", len(processed), len(failures))
        return 1 if failures else 0
    except ExtractionError as exc:
        LOGGER.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
