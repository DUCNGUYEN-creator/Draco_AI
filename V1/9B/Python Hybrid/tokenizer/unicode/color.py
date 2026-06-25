# DracoAI V1 — tokenizer/unicode/color.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""
DracoAI Tokenizer — Color / Style Stripping
============================================
Strips Markdown-style formatting markers, HTML color tags, and other
style annotations that carry no semantic value for the LLM tokenizer.

Mainly used in training pipelines to clean scraped web / chat data.
"""

import re

_HTML_TAG_RE    = re.compile(r"<[^>]+>")
_MD_BOLD_RE     = re.compile(r"\*\*(.+?)\*\*")
_MD_ITALIC_RE   = re.compile(r"\*(.+?)\*|_(.+?)_")
_MD_CODE_RE     = re.compile(r"`{1,3}(.+?)`{1,3}", re.DOTALL)
_MD_HEADING_RE  = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_HTML_ENTITY_RE = re.compile(r"&(?:[a-z]+|#\d+|#x[0-9a-fA-F]+);")

_HTML_ENTITIES = {
    "&amp;": "&", "&lt;": "<", "&gt;": ">",
    "&quot;": '"', "&apos;": "'", "&nbsp;": " ",
}


def strip_html_tags(text: str) -> str:
    """Remove all HTML tags from *text*."""
    return _HTML_TAG_RE.sub("", text)


def decode_html_entities(text: str) -> str:
    """Decode common HTML entities to their Unicode equivalents."""
    for entity, char in _HTML_ENTITIES.items():
        text = text.replace(entity, char)
    return _HTML_ENTITY_RE.sub("", text)


def strip_markdown_formatting(text: str) -> str:
    """
    Strip Markdown bold, italic, headings.
    Code spans/blocks are *kept* (their content is semantically important).
    """
    text = _MD_HEADING_RE.sub("", text)
    text = _MD_BOLD_RE.sub(r"\1", text)
    text = _MD_ITALIC_RE.sub(lambda m: m.group(1) or m.group(2), text)
    return text


def clean_for_training(text: str) -> str:
    """
    Full cleaning pipeline for training corpus text:
    strip HTML tags → decode entities → strip Markdown formatting.
    """
    text = strip_html_tags(text)
    text = decode_html_entities(text)
    text = strip_markdown_formatting(text)
    return text