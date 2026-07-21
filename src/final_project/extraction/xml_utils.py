"""Namespace-independent XML helpers for TEI and JATS."""

from __future__ import annotations

import re
import unicodedata
import xml.etree.ElementTree as ET
from collections.abc import Iterable


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def descendants(element: ET.Element | None, name: str) -> Iterable[ET.Element]:
    if element is None:
        return ()
    return (child for child in element.iter() if local_name(child.tag) == name)


def direct_children(element: ET.Element | None, name: str) -> list[ET.Element]:
    if element is None:
        return []
    return [child for child in element if local_name(child.tag) == name]


def first_descendant(
    element: ET.Element | None, name: str
) -> ET.Element | None:
    return next(iter(descendants(element, name)), None)


def normalize_inline(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    return re.sub(r"\s+", " ", text).strip()


def element_text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return normalize_inline(" ".join(element.itertext()))


def paragraph_text(elements: Iterable[ET.Element]) -> str:
    paragraphs = [element_text(element) for element in elements]
    return "\n\n".join(paragraph for paragraph in paragraphs if paragraph)


def attribute_case_insensitive(element: ET.Element, name: str) -> str:
    for key, value in element.attrib.items():
        if local_name(key).lower() == name.lower():
            return value
    return ""
