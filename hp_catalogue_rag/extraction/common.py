"""Shared helpers for the HP catalogue extraction pipeline."""
from __future__ import annotations

import html
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf  # PyMuPDF

ROOT = Path(__file__).resolve().parent.parent
INPUT_PDF = ROOT / "data" / "input" / "catalogue.pdf"
SPLIT_PDF = ROOT / "data" / "interim" / "catalogue_split.pdf"
INTERIM = ROOT / "data" / "interim"
OUT = ROOT / "output"

SKU_RE = re.compile(r"^[A-Z][A-Z0-9]{4,6}(/A)?$")


# --------------------------------------------------------------------------
# Page geometry: the catalogue is mostly two-page spreads on one PDF page
# --------------------------------------------------------------------------
def iter_half_pages(doc: pymupdf.Document):
    """Yield (pdf_page_index, printed_page_number, page, clip_rect).

    Cover (index 0) and the last page are single pages; everything else is a
    spread whose left/right halves are printed pages 2i and 2i+1.
    """
    for i, pg in enumerate(doc):
        r = pg.rect
        if r.width > r.height * 1.2:
            halves = [
                (2 * i, pymupdf.Rect(0, 0, r.width / 2, r.height)),
                (2 * i + 1, pymupdf.Rect(r.width / 2, 0, r.width, r.height)),
            ]
        else:
            halves = [(1 if i == 0 else 2 * i, r)]
        for printed, clip in halves:
            yield i, printed, pg, clip


# --------------------------------------------------------------------------
# Text lines with superscript (footnote marker) removal
# --------------------------------------------------------------------------
@dataclass
class Span:
    text: str
    x0: float
    x1: float
    y0: float
    y1: float
    size: float
    bold: bool
    color: int
    chars: list = field(default_factory=list)  # [(x0, x1, char)] relative to clip


@dataclass
class Line:
    spans: list[Span] = field(default_factory=list)

    @property
    def x0(self): return min(s.x0 for s in self.spans)
    @property
    def x1(self): return max(s.x1 for s in self.spans)
    @property
    def y0(self): return min(s.y0 for s in self.spans)
    @property
    def y1(self): return max(s.y1 for s in self.spans)
    @property
    def size(self): return max(s.size for s in self.spans)
    @property
    def bold(self): return all(s.bold for s in self.spans)
    @property
    def color(self): return self.spans[0].color

    @property
    def text(self) -> str:
        out = ""
        prev = None
        for s in sorted(self.spans, key=lambda s: s.x0):
            if prev is not None:
                out += "" if s.x0 - prev.x1 < 1.2 else " "
            out += s.text
            prev = s
        return out.strip()


_MARKER = re.compile(r"^[\d,*†‡]{1,3}$")


def get_lines(page: pymupdf.Page, clip: pymupdf.Rect) -> list[Line]:
    """All text lines inside `clip`, x-coordinates relative to clip.x0.

    Footnote reference markers (superscripts such as the '1' in 'Pro1') are
    dropped so cell values stay clean.
    """
    xoff = clip.x0
    lines: list[Line] = []
    data = page.get_text("rawdict", clip=clip)
    for b in data["blocks"]:
        if b["type"] != 0:
            continue
        for l in b["lines"]:
            spans = []
            for s in l["spans"]:
                chars = [(c["bbox"][0] - xoff, c["bbox"][2] - xoff, c["c"]) for c in s["chars"]]
                t = "".join(c[2] for c in chars)
                if not t.strip():
                    continue
                x0, y0, x1, y1 = s["bbox"]
                if not (clip.x0 <= (x0 + x1) / 2 <= clip.x1):
                    continue
                spans.append((s, Span(t, x0 - xoff, x1 - xoff, y0, y1, s["size"],
                                      "Bold" in s["font"], s["color"], chars)))
            if not spans:
                continue
            biggest = max(sp.size for _, sp in spans)
            keep = []
            for raw, sp in spans:
                is_sup = (bool(raw["flags"] & 1) or sp.size < 0.72 * biggest) \
                    and bool(_MARKER.match(sp.text.strip()))
                if not is_sup:
                    keep.append(sp)
            if keep:
                lines.append(Line(keep))
    return lines


# --------------------------------------------------------------------------
# Text cleaning
# --------------------------------------------------------------------------
class Dehyphenator:
    """Repair words broken across lines ('enter-' + 'prise').

    Vocabulary is learned from words that never wrap anywhere in the document,
    so real hyphenated words ('spill-resistant') keep their hyphen.
    """

    def __init__(self, lines_iter):
        self.vocab = Counter()
        for ln in lines_iter:
            for tok in re.findall(r"[A-Za-z][A-Za-z\-]+", ln.text):
                if not tok.endswith("-"):
                    self.vocab[tok.lower()] += 1

    def join(self, parts: list[str]) -> str:
        parts = [p.strip() for p in parts if p and p.strip()]
        if not parts:
            return ""
        out = parts[0]
        for nxt in parts[1:]:
            m = re.search(r"([A-Za-z]+)-$", out)
            n = re.match(r"([A-Za-z]+)", nxt)
            if m and n:
                glued = (m.group(1) + n.group(1)).lower()
                hyph = (m.group(1) + "-" + n.group(1)).lower()
                if self.vocab.get(glued) or (not self.vocab.get(hyph) and nxt[0].islower()):
                    out = out[:-1] + nxt
                    continue
                out = out + nxt
                continue
            out = out + " " + nxt
        return out


def tidy(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", text)  # control chars break Excel
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([;,.)])", r"\1", text)  # gap left by a removed footnote marker
    text = re.sub(r"(\d)\s+([”\"])", r"\1\2", text)
    text = re.sub(r"([”\"])\s+-(\d)", r"\1-\2", text)
    text = text.replace("Finger print", "Fingerprint")
    return text


def norm_name(name: str) -> str:
    """Normalised key used to match product names across sections."""
    n = name.lower().replace("”", '"').replace("“", '"')
    n = re.sub(r"[^a-z0-9]+", "", n)
    return n
