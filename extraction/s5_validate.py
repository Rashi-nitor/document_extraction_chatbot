"""Step 5 - independent checks on the extraction. Writes output/qa_report.md.

1. SKU recount     : SKU-looking tokens found by plain text search inside every
                     Care Pack table vs. rows extracted (must match exactly).
2. Table coverage  : characters of raw text inside each table box vs. characters
                     in the extracted cells (footnote markers explain a small gap).
3. Name matching   : description <-> spec-table products that failed to match.
"""
import json
import re
import sys

import pandas as pd
import pymupdf

from common import INPUT_PDF, OUT, SKU_RE, iter_half_pages, norm_name


def main() -> int:
    tables = json.loads((OUT / "tables_raw.json").read_text(encoding="utf-8"))
    care = pd.read_csv(OUT / "care_packs_long.csv", keep_default_na=False)
    doc = pymupdf.open(INPUT_PDF)
    halves = {printed: (pg, clip) for _, printed, pg, clip in iter_half_pages(doc)}

    problems, lines = 0, ["# Extraction QA report", ""]

    # ---- 1. SKU recount ----------------------------------------------------
    lines += ["## 1. Care Pack SKU recount (source text vs extracted rows)", "",
              "| page | SKUs in source | SKUs extracted | ok |", "|---|---|---|---|"]
    for page in sorted({t["page"] for t in tables if t["kind"] == "care_pack"}):
        pg, clip = halves[page]
        n_src = 0
        for t in tables:
            if t["page"] != page or t["kind"] != "care_pack":
                continue
            x0, y0, x1, y1 = t["bbox"]
            r = pymupdf.Rect(clip.x0 + x0, y0, clip.x0 + x1, y1)
            words = pg.get_text("words", clip=r)
            n_src += sum(1 for w in words if SKU_RE.match(w[4]) and any(c.isdigit() for c in w[4]))
        n_out = int((care.page == page).sum())
        ok = n_src == n_out
        problems += 0 if ok else 1
        lines.append(f"| {page} | {n_src} | {n_out} | {'yes' if ok else '**NO**'} |")

    # ---- 2. coverage ---------------------------------------------------------
    lines += ["", "## 2. Table text coverage", "",
              "| page | kind | source chars | extracted chars | coverage |", "|---|---|---|---|---|"]
    for t in tables:
        pg, clip = halves[t["page"]]
        x0, y0, x1, y1 = t["bbox"]
        r = pymupdf.Rect(clip.x0 + x0, y0, clip.x0 + x1, y1)
        src = len(re.sub(r"\s+", "", pg.get_text("text", clip=r)))
        out = len(re.sub(r"\s+", "", " ".join(
            t["header"] + [c for g in t["groups"] for row in g["rows"] for c in row])))
        cov = out / src if src else 1.0
        flag = "" if cov >= 0.90 else " **check**"
        lines.append(f"| {t['page']} | {t['kind']} | {src} | {out} | {cov:.1%}{flag} |")

    # ---- 3. matching ----------------------------------------------------------
    prods = pd.read_csv(OUT / "products.csv")
    spec = pd.read_csv(OUT / "spec_tables_long.csv", keep_default_na=False)
    pk = {norm_name(n) for n in prods["name"]}
    sk = {norm_name(n): n for n in spec["product"].unique()}
    lines += ["", "## 3. Name matching", "",
              f"- products with description: {len(prods)}",
              f"- products with a spec table: {len(sk)}",
              f"- spec-table products without a description block: "
              f"{', '.join(sorted(n for k, n in sk.items() if k not in pk)) or 'none'}",
              f"- described products without a spec table: "
              f"{', '.join(sorted(n for n in prods['name'] if norm_name(n) not in sk)) or 'none'}"]

    (OUT / "qa_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
