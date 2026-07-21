"""Incremental extraction of a mixed PDF and DOI literature corpus."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from .errors import ExtractionError
from .io import write_document
from .pipeline import ExtractionPipeline


CORPUS_MANIFEST_VERSION = "1.0"
DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)


def normalize_doi(value: str) -> str:
    """Return a stable DOI value suitable for comparison and NCBI lookup."""
    value = value.strip()
    value = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", value, flags=re.I)
    value = re.sub(r"^doi:\s*", "", value, flags=re.I)
    return value.strip().rstrip(".,;").lower()


def read_doi_list(path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read, normalize, validate, and deduplicate one DOI per line."""
    path = Path(path).expanduser().resolve()
    if not path.exists():
        return [], []
    dois: list[str] = []
    failures: list[dict[str, str]] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        doi = normalize_doi(line)
        if not DOI_RE.match(doi):
            failures.append(
                {
                    "source_type": "doi",
                    "source": line,
                    "error": f"Invalid DOI on line {line_number} of {path.name}",
                }
            )
            continue
        if doi not in seen:
            seen.add(doi)
            dois.append(doi)
    return dois, failures


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        for block in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _output_exists(record: dict[str, Any]) -> bool:
    return bool(record.get("output_path")) and Path(record["output_path"]).is_file()


def _document_record(
    document,
    output_path: Path,
    *,
    source_type: str,
    source: str,
    source_key: str,
    fingerprint: str = "",
) -> dict[str, Any]:
    return {
        "source_type": source_type,
        "source": source,
        "source_key": source_key,
        "fingerprint": fingerprint,
        "document_id": document.document_id,
        "output_path": str(output_path.resolve()),
        "extraction_method": document.extraction_method,
        "fallback_used": bool(document.provenance.get("fallback_used")),
        "doi": normalize_doi(str(document.metadata.get("doi", ""))),
        "section_count": len(document.sections),
        "character_count": document.diagnostics.get("character_count", 0),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def _bootstrap_records(output_dir: Path) -> list[dict[str, Any]]:
    """Discover older normalized outputs when no corpus index exists yet."""
    records: list[dict[str, Any]] = []
    for path in sorted((output_dir / "documents").glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        source_type = str(payload.get("source_type", ""))
        source = str(payload.get("source_identifier", ""))
        doi = normalize_doi(str(payload.get("metadata", {}).get("doi", "")))
        fingerprint = ""
        if source_type == "pdf" and Path(source).is_file():
            fingerprint = file_sha256(source)
            source_key = f"pdf:sha256:{fingerprint}"
        elif doi:
            source_key = f"doi:{doi}"
        else:
            continue
        records.append(
            {
                "source_type": "pdf" if source_type == "pdf" else "doi",
                "source": source,
                "source_key": source_key,
                "fingerprint": fingerprint,
                "document_id": payload.get("document_id", path.stem),
                "output_path": str(path.resolve()),
                "extraction_method": payload.get("extraction_method", ""),
                "fallback_used": bool(payload.get("provenance", {}).get("fallback_used")),
                "doi": doi,
                "section_count": len(payload.get("sections", [])),
                "character_count": payload.get("diagnostics", {}).get("character_count", 0),
                "updated_at": payload.get("provenance", {}).get("extracted_at", ""),
            }
        )
    return records


def _load_records(output_dir: Path) -> list[dict[str, Any]]:
    manifest_path = output_dir / "corpus_manifest.json"
    if not manifest_path.is_file():
        return _bootstrap_records(output_dir)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = payload.get("sources", [])
        return records if isinstance(records, list) else []
    except (OSError, json.JSONDecodeError):
        return _bootstrap_records(output_dir)


def _write_corpus_manifest(
    output_dir: Path,
    records: list[dict[str, Any]],
    *,
    processed: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "corpus_manifest.json"
    payload = {
        "schema_version": CORPUS_MANIFEST_VERSION,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "document_count": len({record.get("output_path") for record in records if _output_exists(record)}),
        "sources": sorted(records, key=lambda item: str(item.get("source_key", ""))),
        "last_run": {
            "num_processed": len(processed),
            "num_skipped": len(skipped),
            "num_failures": len(failures),
            "processed": processed,
            "skipped": skipped,
            "failures": failures,
        },
    }
    temporary_path = path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary_path.replace(path)
    return path


@dataclass(slots=True)
class CorpusRunResult:
    manifest_path: Path
    processed: list[dict[str, Any]]
    skipped: list[dict[str, Any]]
    failures: list[dict[str, Any]]


def run_corpus(
    pipeline: ExtractionPipeline,
    *,
    pdf_dir: str | Path,
    doi_list: str | Path,
    output_dir: str | Path,
    recursive: bool = False,
    save_raw_xml: bool = False,
) -> CorpusRunResult:
    """Incrementally extract all PDFs and DOI-only papers into one corpus."""
    pdf_dir = Path(pdf_dir).expanduser().resolve()
    doi_list = Path(doi_list).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    records = _load_records(output_dir)
    by_key = {str(record.get("source_key")): record for record in records}
    processed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    pattern = "**/*.pdf" if recursive else "*.pdf"
    pdf_paths = sorted(pdf_dir.glob(pattern)) if pdf_dir.is_dir() else []
    if not pdf_dir.is_dir():
        failures.append(
            {"source_type": "pdf_directory", "source": str(pdf_dir), "error": "PDF directory does not exist"}
        )

    for pdf_path in pdf_paths:
        fingerprint = file_sha256(pdf_path)
        source_key = f"pdf:sha256:{fingerprint}"
        resolved_pdf = str(pdf_path.resolve())
        replaced_keys = [
            key
            for key, record in by_key.items()
            if record.get("source_type") == "pdf"
            and record.get("source") == resolved_pdf
            and key != source_key
        ]
        for replaced_key in replaced_keys:
            by_key.pop(replaced_key, None)
            for alias_key, alias in list(by_key.items()):
                if alias.get("represented_by") == replaced_key:
                    by_key.pop(alias_key, None)
        existing = by_key.get(source_key)
        if existing and _output_exists(existing):
            existing["source"] = str(pdf_path.resolve())
            skipped.append(
                {"source_type": "pdf", "source": str(pdf_path), "reason": "unchanged_pdf", "document_id": existing.get("document_id", "")}
            )
            continue
        try:
            document = pipeline.extract_pdf(pdf_path)
            document.provenance["source_sha256"] = fingerprint
            document_doi = normalize_doi(str(document.metadata.get("doi", "")))
            represented = next(
                (
                    record
                    for record in by_key.values()
                    if normalize_doi(str(record.get("doi", ""))) == document_doi
                    and document_doi
                    and _output_exists(record)
                ),
                None,
            )
            if represented:
                alias = dict(represented)
                alias.update(
                    {
                        "source_type": "pdf",
                        "source": resolved_pdf,
                        "source_key": source_key,
                        "fingerprint": fingerprint,
                        "represented_by": represented.get("source_key", ""),
                    }
                )
                by_key[source_key] = alias
                skipped.append(
                    {
                        "source_type": "pdf",
                        "source": str(pdf_path),
                        "reason": "doi_already_in_corpus",
                        "document_id": represented.get("document_id", ""),
                    }
                )
                continue
            output_path = write_document(document, output_dir, save_raw_xml=save_raw_xml)
            record = _document_record(
                document,
                output_path,
                source_type="pdf",
                source=resolved_pdf,
                source_key=source_key,
                fingerprint=fingerprint,
            )
            by_key[source_key] = record
            processed.append(record.copy())
        except ExtractionError as exc:
            failures.append({"source_type": "pdf", "source": str(pdf_path), "error": str(exc)})

    records = list(by_key.values())
    doi_to_record = {
        normalize_doi(str(record.get("doi", ""))): record
        for record in records
        if normalize_doi(str(record.get("doi", ""))) and _output_exists(record)
    }
    dois, doi_failures = read_doi_list(doi_list)
    failures.extend(doi_failures)
    for doi in dois:
        source_key = f"doi:{doi}"
        existing = by_key.get(source_key)
        represented = doi_to_record.get(doi)
        if existing and _output_exists(existing):
            skipped.append(
                {"source_type": "doi", "source": doi, "reason": "already_extracted", "document_id": existing.get("document_id", "")}
            )
            continue
        if represented:
            alias = dict(represented)
            alias.update({"source_type": "doi", "source": doi, "source_key": source_key, "represented_by": represented.get("source_key", "")})
            by_key[source_key] = alias
            skipped.append(
                {"source_type": "doi", "source": doi, "reason": "doi_present_in_pdf", "document_id": represented.get("document_id", "")}
            )
            continue
        try:
            document = pipeline.extract_online(doi=doi, allow_abstract_fallback=False)
            output_path = write_document(document, output_dir, save_raw_xml=save_raw_xml)
            record = _document_record(
                document,
                output_path,
                source_type="doi",
                source=doi,
                source_key=source_key,
            )
            by_key[source_key] = record
            if record["doi"]:
                doi_to_record[record["doi"]] = record
            processed.append(record.copy())
        except ExtractionError as exc:
            failures.append({"source_type": "doi", "source": doi, "error": str(exc)})

    manifest_path = _write_corpus_manifest(
        output_dir,
        list(by_key.values()),
        processed=processed,
        skipped=skipped,
        failures=failures,
    )
    return CorpusRunResult(manifest_path, processed, skipped, failures)
