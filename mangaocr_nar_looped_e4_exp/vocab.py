"""Output vocabulary: 6,817 characters, id 6817 = end of text."""
from __future__ import annotations

import json
from pathlib import Path


class Vocab:
    def __init__(self, path: Path):
        self.chars = json.loads(Path(path).read_text(encoding="utf-8"))["characters"]
        self.eos = len(self.chars)

    def decode(self, rows) -> list[str]:
        """rows: iterable of id sequences (65 per crop) -> texts, cut at the first end-of-text id."""
        out = []
        for row in rows:
            row = list(row)
            row = row[: row.index(self.eos)] if self.eos in row else row
            out.append("".join(self.chars[i] for i in row if i < self.eos))
        return out
