"""Step 2 (optional) - Unstructured.io extraction on the split pages.

This is the layout-model route you asked about. It produces a typed element
list (Title / NarrativeText / Table / Image ...) that is handy for RAG, and a
second opinion on table detection. The authoritative structured CSV/Excel
outputs come from s3_native_extract.py, which reads the PDF's own table rules.

Needs system tools (poppler + tesseract) and, on first run, internet access to
download the layout model. Run:

    python extraction/s2_unstructured_hi_res.py                    # hi_res, all pages
    python extraction/s2_unstructured_hi_res.py --pages 10 11 12   # quick test
    python extraction/s2_unstructured_hi_res.py --strategy fast    # no models, no tables

Outputs (output/unstructured/):
    elements.json    every element with page, category, text, coordinates
    tables/          one HTML + CSV per detected table
    images/          cropped images/tables (hi_res only)
    chunks.jsonl     title-based chunks (max 1500 chars) for RAG / embeddings
"""
import argparse
import json
from io import StringIO

import pandas as pd

from common import OUT, SPLIT_PDF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="hi_res", choices=["hi_res", "fast", "ocr_only"])
    ap.add_argument("--pages", type=int, nargs="*", help="printed page numbers (default: all)")
    ap.add_argument("--model", default=None, help="hi_res layout model, e.g. yolox")
    args = ap.parse_args()

    from unstructured.chunking.title import chunk_by_title
    from unstructured.partition.pdf import partition_pdf
    from unstructured.staging.base import convert_to_dict

    out = OUT / "unstructured"
    (out / "tables").mkdir(parents=True, exist_ok=True)

    src = SPLIT_PDF
    if args.pages:  # build a small PDF with just the requested pages
        import pymupdf
        full, sub = pymupdf.open(SPLIT_PDF), pymupdf.open()
        for p in args.pages:
            sub.insert_pdf(full, from_page=p - 1, to_page=p - 1)
        src = out / "subset.pdf"
        sub.save(src)

    kwargs = dict(filename=str(src), strategy=args.strategy, languages=["eng"])
    if args.strategy == "hi_res":
        kwargs.update(infer_table_structure=True,
                      extract_image_block_types=["Image", "Table"],
                      extract_image_block_output_dir=str(out / "images"))
        if args.model:
            kwargs["hi_res_model_name"] = args.model
    elements = partition_pdf(**kwargs)

    page_map = args.pages or None
    rows = convert_to_dict(elements)
    for r in rows:
        n = r["metadata"].get("page_number")
        if page_map and n:
            r["metadata"]["printed_page"] = page_map[n - 1]
        else:
            r["metadata"]["printed_page"] = n
    (out / "elements.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False),
                                       encoding="utf-8")

    n_tables = 0
    for i, el in enumerate(elements):
        if el.category != "Table":
            continue
        n_tables += 1
        page = el.metadata.page_number
        html = getattr(el.metadata, "text_as_html", None)
        if not html:
            continue
        (out / "tables" / f"p{page:02d}_t{i:03d}.html").write_text(html, encoding="utf-8")
        try:
            pd.read_html(StringIO(html))[0].to_csv(out / "tables" / f"p{page:02d}_t{i:03d}.csv",
                                                   index=False)
        except ValueError:
            pass

    chunks = chunk_by_title(elements, max_characters=1500, combine_text_under_n_chars=200)
    with open(out / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({"page": c.metadata.page_number, "type": c.category,
                                "text": c.text}, ensure_ascii=False) + "\n")

    counts = pd.Series([e.category for e in elements]).value_counts().to_dict()
    print(f"{len(elements)} elements, {n_tables} tables, {len(chunks)} chunks -> {out}")
    print(counts)


if __name__ == "__main__":
    main()
