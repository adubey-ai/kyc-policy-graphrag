"""Shared tokenisation for sparse retrieval."""

from __future__ import annotations

import re

TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_+.-]*")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN.findall(text)]
