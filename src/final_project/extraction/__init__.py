"""Scientific-paper extraction interfaces."""

from .models import ExtractedDocument, Section
from .pipeline import ExtractionPipeline

__all__ = ["ExtractedDocument", "ExtractionPipeline", "Section"]
