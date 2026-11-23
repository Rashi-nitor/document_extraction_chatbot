"""Step 3 - structure-aware extraction straight from the PDF.

The catalogue is a *digital* PDF: every table is drawn with horizontal rules
and every character carries font, size and position. We use that instead of
guessing layout from pixels:

  * table rules      -> row boundaries and column starts
  * bold spans       -> row labels, product names, column headers
  * x-position       -> which column a cell belongs to (so empty Care Pack
                        cells never shift SKUs into the wrong column)
  * superscript flag -> footnote markers removed ('Pro1' -> 'Pro')

Outputs (output/):
  products.csv / products.json     name + description per product
  features_and_captions.csv        bullet feature blocks, accessory captions
  spec_tables_long.csv             one row per product x attribute
  care_packs_long.csv              one row per SKU
  care_pack_recommended.csv        recommended service descriptions
  toc.csv, pages.jsonl, tables_raw.json
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import pandas as pd
import pymupdf

from common import (INPUT_PDF, OUT, SKU_RE, Dehyphenator, Line, get_lines,
                    iter_half_pages, tidy)

BOILERPLATE_HEAD = re.compile(
    r"^(focus on your business|hp care pack services|warranty upgrade|"
    r"enhanced protection|table of contents|hp recommends windows)", re.I)
BOILERPLATE_BODY = re.compile(
    r"(please check hp care pack availability|hp care pack services make buying)", re.I)
BAND = 48  # pt above the first rule that still counts as table header


# ----------------------------------------------------------------------------
# Table detection
# ----------------------------------------------------------------------------
@dataclass
class Table:
    ys: list[float]
    level_starts: dict[float, list[float]]
    starts: list[float]
    x_max: float
    first_cover: list[tuple[float, float]]
    bottom: float = 0.0
    top: float = 0.0
    kind: str = "generic"
    header: list[str] = field(default_factory=list)
    groups: list[dict] = field(default_factory=list)


def _cluster(values, tol=3.0):
    out: list[float] = []
    for v in sorted(values):
        if not out or v - out[-1] > tol:
            out.append(v)
    return out


def _near(v, arr, tol=3.0):
    return any(abs(v - a) <= tol for a in arr)


def find_tables(page, clip) -> list[Table]:
    xoff = clip.x0
    levels: dict[float, list[tuple[float, float]]] = {}
    for d in page.get_drawings():
        r = d["rect"]
        if r.height > 1.6 or r.width < 18:
            continue
        if r.x0 < clip.x0 - 1 or r.x1 > clip.x1 + 1:
            continue
        y = round((r.y0 + r.y1) / 2, 1)
        levels.setdefault(y, []).append((r.x0 - xoff, r.x1 - xoff))
    # merge levels that differ by <1pt
    ys_sorted = sorted(levels)
    merged: dict[float, list] = {}
    for y in ys_sorted:
        key = next((k for k in merged if abs(k - y) < 1.0), None)
        if key is None:
            merged[y] = list(levels[y])
        else:
            merged[key].extend(levels[y])

    tables: list[Table] = []
    cur: Table | None = None
    for y in sorted(merged):
        segs = merged[y]
        starts = _cluster([s[0] for s in segs])
        compatible = False
        if cur is not None:
            cur_starts = cur.starts
            new_in_cur = all(_near(s, cur_starts) for s in starts)
            cur_in_new = all(_near(s, starts) for s in cur_starts)
            compatible = new_in_cur or cur_in_new
        if cur is None or not compatible:
            if cur is not None:
                tables.append(cur)
            cur = Table(ys=[y], level_starts={y: starts}, starts=starts,
                        x_max=max(s[1] for s in segs), first_cover=sorted(segs))
        else:
            cur.ys.append(y)
            cur.level_starts[y] = starts
            cur.starts = _cluster(cur.starts + starts)
            cur.x_max = max(cur.x_max, max(s[1] for s in segs))
    if cur is not None:
        tables.append(cur)
    return [t for t in tables if len(t.ys) >= 2 and len(t.starts) >= 2]


def col_of(x0: float, starts: list[float]) -> int | None:
    idx = None
    for j, s in enumerate(starts):
        if x0 >= s - 2.5:
            idx = j
    return idx


def cell_text(lines: list[Line], dh: Dehyphenator) -> str:
    """Join the lines of one cell (already filtered) into clean text."""
    if not lines:
        return ""
    lines = sorted(lines, key=lambda l: (round(l.y0 / 2), l.x0))
    return tidy(dh.join([l.text for l in lines]))


def split_to_cells(lines_or_spans, starts, dh):
    """Assign single-span pseudo-lines to columns; returns {col: text}."""
    by_col: dict[int, list[Line]] = {}
    for ln in lines_or_spans:
        j = col_of(ln.x0, starts)
        if j is not None:
            by_col.setdefault(j, []).append(ln)
    return {j: cell_text(v, dh) for j, v in by_col.items()}


def explode_lines(lines: list[Line], starts: list[float]) -> list[Line]:
    """One Line per (span, column).

    A PDF text run can straddle two table columns (e.g. two product names set
    in one run), so spans are cut at character level wherever the column
    changes.
    """
    from common import Span
    out = []
    for ln in lines:
        for sp in ln.spans:
            if not sp.chars:
                out.append(Line([sp]))
                continue
            piece, cur_col = [], None
            for ch in sp.chars:
                j = col_of(ch[0], starts) if ch[2].strip() else cur_col
                if cur_col is not None and j != cur_col and piece:
                    out.append(_mk(piece, sp))
                    piece = []
                cur_col = j if j is not None else cur_col
                piece.append(ch)
            if piece:
                out.append(_mk(piece, sp))
    return [l for l in out if l.spans[0].text.strip()]


def _mk(chars, sp):
    from common import Span
    text = "".join(c[2] for c in chars)
    lead = len(text) - len(text.lstrip())
    chars = chars[lead:]
    text = text.strip()
    return Line([Span(text, chars[0][0], chars[-1][1], sp.y0, sp.y1, sp.size, sp.bold,
                      sp.color, chars)]) if chars else Line([Span("", 0, 0, 0, 0, 0, False, 0)])


def parse_table(t: Table, lines: list[Line], dh: Dehyphenator):
    ys = t.ys
    starts = t.starts
    pieces = explode_lines(lines, starts)
    ncol = len(starts)

    # ---- header band: text just above the first rule, in columns it covers
    covered = t.first_cover
    band = [p for p in pieces
            if ys[0] - BAND <= (p.y0 + p.y1) / 2 < ys[0]
            and any(a - 2.5 <= p.x0 <= b for a, b in covered)
            and ((p.size >= 6.5 and p.bold) or p.text in ('"', '”'))
            and not re.match(r'^(hp\.com.*|c?pc|c)$', p.text.strip())]
    hdr = split_to_cells(band, starts, dh)
    t.header = [hdr.get(j, "") for j in range(ncol)]
    t.top = min([(p.y0) for p in band], default=ys[0])

    # ---- row intervals; the last row has no closing rule -> chain downward
    bounds = list(ys)
    tail = sorted({round(p.y0, 1) for p in pieces if p.y0 >= ys[-1] - 1 and p.size >= 6.5
                   and p.x0 >= starts[0] - 2.5 and p.x0 <= t.x_max})
    end = ys[-1]
    if tail and tail[0] - ys[-1] <= 12:
        prev = tail[0]
        end = prev + 9
        for y in tail[1:]:
            if y - prev <= 11.5:
                prev, end = y, y + 9
            else:
                break
        bounds.append(end + 3)
    t.bottom = bounds[-1]

    rows = []  # (level_y, [col texts])
    for i in range(len(bounds) - 1):
        ya, yb = bounds[i], bounds[i + 1]
        inside = [p for p in pieces if ya - 0.5 <= (p.y0 + p.y1) / 2 < yb - 0.5
                  and p.size >= 6.5 and p.x0 >= starts[0] - 2.5 and p.x0 <= t.x_max]
        cells = split_to_cells(inside, starts, dh)
        rows.append((ya, [cells.get(j, "") for j in range(ncol)]))

    # ---- group rows: a rule that spans column 0 starts a new group
    groups: list[dict] = []
    for ya, cells in rows:
        lvl = t.level_starts.get(ya, [])
        has_c0 = any(abs(s - starts[0]) <= 3 for s in lvl)
        if has_c0 or not groups:
            groups.append({"rows": []})
        groups[-1]["rows"].append(cells)
    for g in groups:
        g["label"] = tidy(" ".join(r[0] for r in g["rows"] if r[0]))
    t.groups = groups

    # ---- classify
    body = [c for g in groups for r in g["rows"] for c in r[2:] if c]
    sku_ratio = (sum(bool(SKU_RE.match(c)) for c in body) / len(body)) if body else 0
    head_text = " ".join(t.header).lower()
    if sku_ratio > 0.5:
        t.kind = "care_pack"
    elif "recommended" in head_text and "description" in head_text:
        t.kind = "recommended"
    elif ncol >= 3:
        t.kind = "spec"
    return t


# ----------------------------------------------------------------------------
# Section / page level text
# ----------------------------------------------------------------------------
def page_title(lines: list[Line]) -> str:
    def ok(l):
        return l.bold and not BOILERPLATE_HEAD.match(l.text)
    top = sorted([l for l in lines if ok(l) and l.y0 < 75 and l.size >= 13],
                 key=lambda l: (l.y0, l.x0))
    blue = [l for l in top if l.color == 0xA1DD]
    top = blue or top
    if top:
        return tidy(" ".join(l.text for l in top))
    big = sorted([l for l in lines if ok(l) and l.size >= 20 and l.y0 < 160],
                 key=lambda l: (l.y0, l.x0))
    return tidy(" ".join(l.text for l in big))


def extract_products(lines: list[Line], dh: Dehyphenator, table_boxes, printed, section):
    """Pair bold headings with the body text underneath them (same column)."""
    def in_table(l: Line):
        return any(b[0] <= (l.y0 + l.y1) / 2 <= b[1] for b in table_boxes)

    content = [l for l in lines if not in_table(l) and l.y0 < 795 and l.size > 6.5]
    # merge consecutive bold lines into one heading (wrapped names)
    heads, used = [], set()
    bold_lines = sorted([l for l in content if l.bold and 7.5 <= l.size <= 15
                         and not (l.y0 < 75 and l.size >= 13 and l.color == 0xA1DD)
                         and not BOILERPLATE_HEAD.match(l.text)],
                        key=lambda l: (round(l.x0 / 15), l.y0))
    i = 0
    while i < len(bold_lines):
        cur = [bold_lines[i]]
        j = i + 1
        while j < len(bold_lines):
            a, b = cur[-1], bold_lines[j]
            if (abs(a.x0 - b.x0) <= 3 and 0 <= b.y0 - a.y1 <= 3.5
                    and abs(a.size - b.size) < 0.5):
                cur.append(b)
                j += 1
            else:
                break
        heads.append(cur)
        i = j
    for h in heads:
        for l in h:
            used.add(id(l))

    # ignore title lines that slipped through
    heads = [h for h in heads if not (h[0].y0 < 75 and h[0].size >= 13)]
    body_lines = [l for l in content if id(l) not in used and not l.bold]

    results = []
    for h in heads:
        hx, hy = h[0].x0, h[-1].y1
        same_col_heads_below = [o[0].y0 for o in heads
                                if o is not h and abs(o[0].x0 - hx) <= 12 and o[0].y0 > hy - 1]
        limit = min(same_col_heads_below) if same_col_heads_below else 10_000
        body = sorted([b for b in body_lines
                       if abs(b.x0 - hx) <= 12 and hy - 1 <= b.y0 < limit
                       and b.y0 - hy < 260],
                      key=lambda b: b.y0)
        # stop at a large vertical gap (next unrelated element)
        kept, last = [], hy
        for b in body:
            if b.y0 - last > 26:
                break
            kept.append(b)
            last = b.y1
        name = tidy(dh.join([l.text for l in h]))
        desc = tidy(dh.join([b.text for b in kept]))
        if BOILERPLATE_BODY.search(desc):
            continue
        size, color = h[0].size, h[0].color
        if color == 0xA1DD or desc.startswith("•"):
            kind = "feature_group"
        elif size < 9.5:
            kind = "caption"
        else:
            kind = "product"
        results.append(dict(page=printed, section=section, name=name,
                            description=desc, kind=kind, col_x=round(hx), row_y=round(h[0].y0)))
    return results


# ----------------------------------------------------------------------------
def main():
    OUT.mkdir(exist_ok=True)
    doc = pymupdf.open(INPUT_PDF)
    pages = list(iter_half_pages(doc))
    all_lines = {printed: get_lines(pg, clip) for _, printed, pg, clip in pages}
    dh = Dehyphenator(l for ls in all_lines.values() for l in ls)

    products, tables_raw, page_rows = [], [], []
    spec_long, care_long, rec_rows, toc = [], [], [], []

    for idx, printed, pg, clip in pages:
        lines = all_lines[printed]
        section = page_title(lines)
        tables = [parse_table(t, lines, dh) for t in find_tables(pg, clip)]
        boxes = [(t.top - 1, t.bottom + 1) for t in tables]

        foot = tidy(" ".join(l.text for l in sorted(lines, key=lambda l: (l.y0, l.x0))
                             if l.size <= 6.2 and l.y0 < 830))
        intro = tidy(dh.join([l.text for l in sorted(lines, key=lambda l: (l.y0, l.x0))
                              if not l.bold and l.size >= 12]))
        if printed == 1:
            section = "Product catalogue 2014"
        page_rows.append(dict(page=printed, pdf_page=idx + 1, section=section,
                              intro=intro, footnotes=foot, n_tables=len(tables)))

        # ---- table of contents
        if printed == 4:
            for l in lines:
                m = re.match(r"^(\d+)\s*-\s*(\d+)$", l.text.replace("\t", " ").strip())
                if m:
                    names = [o for o in lines if abs(o.y0 - l.y0) < 4 and o.x0 > l.x1]
                    if names:
                        toc.append(dict(start_page=int(m[1]), end_page=int(m[2]),
                                        section=tidy(names[0].text)))

        # ---- tables
        for ti, t in enumerate(tables):
            raw = dict(page=printed, section=section, table_index=ti, kind=t.kind,
                       bbox=[round(t.starts[0] - 2.5, 1), round(t.top - 1, 1),
                             round(t.x_max + 1, 1), round(t.bottom, 1)],
                       header=t.header,
                       groups=[dict(label=g["label"], rows=g["rows"]) for g in t.groups])
            tables_raw.append(raw)

            if t.kind == "spec":
                names = {j: tidy(h) for j, h in enumerate(t.header) if j >= 1 and h.strip()}
                for g in t.groups:
                    row = g["rows"][0]
                    attr = g["label"]
                    if not attr or attr.lower() == "specifications":
                        continue
                    for j, prod in names.items():
                        val = " ".join(r[j] for r in g["rows"] if r[j]).strip()
                        spec_long.append(dict(page=printed, section=section, table_index=ti,
                                              product=prod, attribute=attr, value=val))

            elif t.kind == "care_pack":
                fam = {j: tidy(h) for j, h in enumerate(t.header) if j >= 2 and h.strip()}
                base = {}
                for g in t.groups:
                    for r in g["rows"]:
                        if r[1].lower().startswith("parts/labour"):
                            base = {j: r[j] for j in fam if r[j]}
                level = None
                for g in t.groups:
                    for r in g["rows"]:
                        if r[1].lower().startswith("parts/labour") or not any(r[2:]):
                            continue
                        lvl = g["label"] or level
                        level = lvl
                        for j, family in fam.items():
                            sku = r[j]
                            if sku and SKU_RE.match(sku):
                                care_long.append(dict(
                                    page=printed, section=section, product_family=family,
                                    base_warranty_p_l_o=base.get(j, ""),
                                    service_level=lvl, term=r[1], sku=sku))

            elif t.kind == "recommended":
                for g in t.groups:
                    r = g["rows"][0]
                    if len(r) >= 3 and (r[0] or r[1]):
                        rec_rows.append(dict(page=printed, section=section,
                                             service=g["label"], description=r[1],
                                             options=" ".join(x[2] for x in g["rows"] if x[2])))

        products += extract_products(lines, dh, boxes, printed, section)

    # ------------------------------------------------------------------ save
    prod = pd.DataFrame(products)
    prod_main = prod[(prod.kind == "product") & (prod.description.str.len() > 0)]
    prod_main = prod_main.drop(columns=["kind"]).reset_index(drop=True)
    prod_other = prod[prod.kind != "product"].reset_index(drop=True)
    prod_main.to_csv(OUT / "products.csv", index=False)
    prod_main.to_json(OUT / "products.json", orient="records", indent=2, force_ascii=False)
    prod_other.to_csv(OUT / "features_and_captions.csv", index=False)
    pd.DataFrame(spec_long).to_csv(OUT / "spec_tables_long.csv", index=False)
    pd.DataFrame(care_long).to_csv(OUT / "care_packs_long.csv", index=False)
    pd.DataFrame(rec_rows).to_csv(OUT / "care_pack_recommended.csv", index=False)
    pd.DataFrame(toc).to_csv(OUT / "toc.csv", index=False)
    with open(OUT / "pages.jsonl", "w", encoding="utf-8") as f:
        for r in page_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (OUT / "tables_raw.json").write_text(json.dumps(tables_raw, indent=1, ensure_ascii=False),
                                         encoding="utf-8")
    print(f"products={len(prod_main)} other={len(prod_other)} spec_rows={len(spec_long)} "
          f"care_pack_skus={len(care_long)} recommended={len(rec_rows)} tables={len(tables_raw)}")


if __name__ == "__main__":
    main()
