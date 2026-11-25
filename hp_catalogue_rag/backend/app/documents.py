"""Turn the extracted catalogue data (output/*.csv) into retrieval chunks.

A RAG chatbot is only as good as its chunks. Instead of cutting the text every
N characters, each chunk here is one *meaningful unit* of the catalogue, and
every chunk starts with the product / table it belongs to so it makes sense on
its own:

  overview   one product: name + marketing description
  specs      one product: all spec attributes (+ a few normalised quick facts)
  care_pack  one Care Pack table block: family x service level -> term -> SKU,
             plus the list of products that family applies to
  service    recommended Care Pack service descriptions
  feature    feature groups / accessory blurbs
  notes      per-page footnotes
  catalogue  table of contents (how the catalogue is organised)
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

MAX_CHUNK_CHARS = 1800


@dataclass
class Chunk:
    id: str
    text: str
    doc_type: str
    category: str = ""
    page: int | None = None
    product: str = ""
    title: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


# ----------------------------------------------------------------------------
def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, keep_default_na=False)


def _clean(v) -> str:
    return re.sub(r"\s+", " ", str(v)).strip()


def _split_lines(header: str, lines: list[str], limit: int = MAX_CHUNK_CHARS) -> list[str]:
    """Pack `lines` into blocks <= limit chars, repeating `header` on each block."""
    blocks, cur = [], []
    size = len(header)
    for ln in lines:
        if cur and size + len(ln) + 1 > limit:
            blocks.append(header + "\n" + "\n".join(cur))
            cur, size = [], len(header)
        cur.append(ln)
        size += len(ln) + 1
    if cur:
        blocks.append(header + "\n" + "\n".join(cur))
    return blocks


class CategoryMap:
    """Page -> catalogue category, from the table of contents.

    The TOC is more reliable than page headers (page 24 is printed with the
    header 'HP tablets' although it belongs to the desktops section).
    """

    def __init__(self, toc: pd.DataFrame):
        self.ranges = [(int(r.start_page), int(r.end_page), _clean(r.section))
                       for r in toc.itertuples()] if len(toc) else []

    def __call__(self, page: int | None, fallback: str = "") -> str:
        if page is None:
            return fallback
        for a, b, name in self.ranges:
            if a <= page <= b:
                return name
        return fallback


# ----------------------------------------------------------------------------
# Which products does a Care Pack column apply to?
# ----------------------------------------------------------------------------
def family_products(family: str, names: list[str]) -> list[str]:
    """Best-effort mapping of a Care Pack family label to product names.

    'ProBook 4xx (s+t series)'  -> HP ProBook 430/450/455/470 G2
    'ProDesk/ProOne 4xx ...'    -> every ProDesk 4xx / ProOne 4xx
    'WorkStation Zxxx'          -> HP Z230 / Z420 / Z620 / Z820 / Z1 ...
    'ElitePad'                  -> HP ElitePad 1000 G2
    Labels that cannot be mapped (monitor size classes) return [].
    """
    f = family.lower()
    hits: set[str] = set()
    used_series = False

    # numeric series: "<name>[/<name>] Nxx"
    for m in re.finditer(r"((?:[a-z0-9]+\s*/\s*)*[a-z]+(?:\s[a-z]+)?)\s+(\d)xx", f):
        used_series = True
        for alt in (a.strip() for a in m.group(1).split("/")):
            pat = re.compile(rf"\b{re.escape(alt)}\s+{m.group(2)}\d\d\b")
            hits |= {n for n in names if pat.search(n.lower())}

    # letter series: "Zxxx"
    for m in re.finditer(r"\b([a-z])xxx\b", f):
        used_series = True
        pat = re.compile(rf"\b{m.group(1)}\d{{1,4}}\b")
        hits |= {n for n in names if pat.search(n.lower())}

    # plain names ("EliteBook / Zbook", "ElitePad", "Thin Clients") - only when the
    # label has no numeric series, otherwise "ProDesk" would also pull in ProDesk 600
    if not used_series:
        parens = re.findall(r"\(([^)]*)\)", f)
        sources = [re.sub(r"\([^)]*\)", " ", f)] + [p for p in parens if "series" not in p]
        for src in sources:
            for tok in (t.strip() for t in src.split("/")):
                if len(tok) < 5:
                    continue
                singular = tok[:-1] if tok.endswith("s") else tok
                hits |= {n for n in names if singular in n.lower()}

    return sorted(hits)


# ----------------------------------------------------------------------------
def build_chunks(data_dir: Path) -> list[Chunk]:
    data_dir = Path(data_dir)
    toc = _read(data_dir / "toc.csv")
    cat = CategoryMap(toc)
    chunks: list[Chunk] = []

    products = _read(data_dir / "products.csv")
    master = _read(data_dir / "products_master.csv")
    spec = _read(data_dir / "spec_tables_long.csv")
    care = _read(data_dir / "care_packs_long.csv")
    recommended = _read(data_dir / "care_pack_recommended.csv")
    extras = _read(data_dir / "features_and_captions.csv")

    # -- overview chunks ---------------------------------------------------------
    for r in products.itertuples():
        desc = _clean(r.description)
        if not desc:
            continue
        page = int(r.page)
        category = cat(page, _clean(r.section))
        text = (f"{_clean(r.name)}\nCategory: {category} (catalogue page {page})\n"
                f"Description: {desc}")
        chunks.append(Chunk(f"overview-{page}-{len(chunks)}", text, "overview", category, page,
                            _clean(r.name), _clean(r.name)))

    # -- spec chunks ------------------------------------------------------------
    facts = {}
    if len(master):
        for r in master.itertuples():
            bits = []
            if r.weight_kg != "":
                bits.append(f"weight {float(r.weight_kg):g} kg")
            if r.display_inch != "":
                bits.append(f"display {float(r.display_inch):g} inch ({float(r.display_cm):g} cm)")
            if r.max_memory_gb != "":
                bits.append(f"max memory {float(r.max_memory_gb):g} GB")
            if bits:
                facts[_clean(r.name)] = "; ".join(bits)

    for (product, page), g in spec.groupby(["product", "page"], sort=False):
        page = int(page)
        product = _clean(product)
        category = cat(page, _clean(g["section"].iloc[0]))
        header = f"{product} - specifications\nCategory: {category} (catalogue page {page})"
        if product in facts:
            header += f"\nQuick facts: {facts[product]}"
        lines = [f"- {_clean(a)}: {_clean(v)}" for a, v in zip(g["attribute"], g["value"])
                 if _clean(v)]
        for i, block in enumerate(_split_lines(header, lines)):
            chunks.append(Chunk(f"specs-{page}-{len(chunks)}", block, "specs", category, page,
                                product, f"{product} specifications"))

    # -- comparison chunks: key specs of EVERY model in a category -------------------------
    # Questions such as "which workstation has the most memory?" need all models at once,
    # which no single specs chunk can answer.
    KEY_ATTRS = ["Weight", "Display size (diagonal)", "Displays size (diagonal)", "Display",
                 "Resolution", "Brightness", "Contrast ratio", "Response time", "Power consumption",
                 "Processors", "Processor speed", "Memory", "Memory upgrade", "Memory slots",
                 "Graphics", "Storage", "Internal drive bays", "Form factor"]
    if len(spec):
        sp = spec.assign(_cat=[cat(int(p), _clean(sec)) for p, sec in zip(spec["page"], spec["section"])])
        for category, g in sp.groupby("_cat", sort=False):
            lines = []
            for product, gp in g.groupby("product", sort=False):
                vals = {a: _clean(v) for a, v in zip(gp["attribute"], gp["value"]) if _clean(v)}
                bits = [f"{a}: {vals[a][:110] + ('…' if len(vals[a]) > 110 else '')}"
                        for a in KEY_ATTRS if a in vals]
                page = int(gp["page"].iloc[0])
                lines.append(f"- {_clean(product)} (page {page}) | " + " | ".join(bits))
            header = (f"Comparison table - {category}: key specifications of every model "
                      f"(abridged; see each model's specifications for full detail)")
            for block in _split_lines(header, lines, 1900):
                chunks.append(Chunk(f"compare-{len(chunks)}", block, "comparison", category,
                                    int(g["page"].min()), "", f"{category} comparison"))

    # -- care pack chunks ------------------------------------------------------------
    spec_names = sorted({_clean(p) for p in spec["product"]}) if len(spec) else []
    if len(care):
        for (page, family, level), g in care.groupby(["page", "product_family", "service_level"],
                                                     sort=False):
            page = int(page)
            category = cat(page, _clean(g["section"].iloc[0]))
            applies = family_products(_clean(family), spec_names)
            base = _clean(g["base_warranty_p_l_o"].iloc[0])
            head = [f"HP Care Pack - {category}",
                    f"Product family: {_clean(family)}",
                    f"Service level: {_clean(level)}",
                    f"Base warranty (parts/labour/onsite years): {base}",
                    f"Catalogue page {page}"]
            if applies:
                head.append("Applies to: " + ", ".join(applies))
            lines = [f"- {_clean(t)}: SKU {_clean(s)}" for t, s in zip(g["term"], g["sku"])]
            text = "\n".join(head) + "\n" + "\n".join(lines)
            chunks.append(Chunk(f"care-{page}-{len(chunks)}", text, "care_pack", category, page,
                                "", f"Care Pack {family} / {level}",
                                {"family": _clean(family), "skus": [_clean(s) for s in g["sku"]]}))

    # -- recommended services -----------------------------------------------------------
    for r in recommended.itertuples():
        page = int(r.page)
        category = cat(page, _clean(r.section))
        text = (f"HP Care Pack recommended service: {_clean(r.service)}\n"
                f"Category: {category} (catalogue page {page})\n"
                f"Description: {_clean(r.description)}\nTerm options: {_clean(r.options)}")
        chunks.append(Chunk(f"service-{page}-{len(chunks)}", text, "service", category, page, "",
                            _clean(r.service)))

    # -- feature groups / accessory blurbs -------------------------------------------------
    for r in extras.itertuples():
        desc = _clean(r.description)
        if not desc or desc == _clean(r.name):
            continue
        page = int(r.page)
        category = cat(page, _clean(r.section))
        text = f"{_clean(r.name)} ({category}, catalogue page {page})\n{desc}"
        chunks.append(Chunk(f"feature-{page}-{len(chunks)}", text, "feature", category, page, "",
                            _clean(r.name)))

    # -- page footnotes ---------------------------------------------------------------------
    pages_path = data_dir / "pages.jsonl"
    if pages_path.exists():
        for line in pages_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            note = _clean(row.get("footnotes", ""))
            note = re.sub(r"Product image may differ from actual product\.?", "", note).strip()
            if len(note) < 40:
                continue
            page = int(row["page"])
            category = cat(page, _clean(row.get("section", "")))
            chunks.append(Chunk(f"notes-{page}", f"Footnotes on catalogue page {page} ({category}): {note}",
                                "notes", category, page, "", f"Footnotes p.{page}"))

    # -- table of contents ---------------------------------------------------------------------
    if len(toc):
        text = "Catalogue structure (HP PC Catalogue 2014):\n" + "\n".join(
            f"- {_clean(r.section)}: pages {r.start_page}-{r.end_page}" for r in toc.itertuples())
        chunks.append(Chunk("catalogue-toc", text, "catalogue", "Catalogue", 4, "", "Table of contents"))

    # unique ids
    seen: dict[str, int] = {}
    for c in chunks:
        n = seen.get(c.id, 0)
        seen[c.id] = n + 1
        if n:
            c.id = f"{c.id}-{n}"
    return chunks


if __name__ == "__main__":  # quick look: python -m app.documents
    from .config import get_settings

    cs = build_chunks(get_settings().data_dir)
    from collections import Counter

    print(len(cs), Counter(c.doc_type for c in cs))
