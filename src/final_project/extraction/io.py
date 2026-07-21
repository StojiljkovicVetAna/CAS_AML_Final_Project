"""Filesystem output helpers for normalized extraction records."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

from .models import ExtractedDocument


def safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned or "document"


def write_document(
    document: ExtractedDocument,
    output_dir: str | Path,
    *,
    save_raw_xml: bool = False,
) -> Path:
    output_dir = Path(output_dir).expanduser().resolve()
    document_dir = output_dir / "documents"
    document_dir.mkdir(parents=True, exist_ok=True)
    filename = safe_filename(document.document_id)
    output_path = document_dir / f"{filename}.json"
    output_path.write_text(
        json.dumps(document.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if save_raw_xml and document.raw_xml:
        raw_dir = output_dir / "raw_xml"
        raw_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / f"{filename}.xml").write_text(document.raw_xml, encoding="utf-8")
    return output_path


def write_manifest(
    output_dir: str | Path,
    *,
    command: str,
    processed: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> Path:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "manifest.json"
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "num_processed": len(processed),
        "num_failures": len(failures),
        "processed": processed,
        "failures": failures,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path
