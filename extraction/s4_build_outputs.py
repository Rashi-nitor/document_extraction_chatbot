"""Step 4 - merge, normalise and package the extracted data.

Reads the CSVs written by step 3 and produces:
  output/spec_tables_wide.csv     one row per product, one column per attribute
  output/products_master.csv      description + normalised specs, joined by name
  output/care_packs_long.csv      (rewritten with cleaned family names)
  output/catalogue_extracted.xlsx every table above in one workbook
"""
import re

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from common import OUT, norm_name


def balance_parens(s: str) -> str:
    return s + ")" * (s.count("(") - s.count(")")) if isinstance(s, str) else s


def first_number(text: str, unit_pat: str):
    if not isinstance(text, str):
        return None
    m = re.search(rf"(\d+(?:[.,]\d+)?)\s*{unit_pat}", text)
    return float(m.group(1).replace(",", ".")) if m else None


def max_memory_gb(text: str):
    """'Upgradeable to 16,384 MB' -> 16 ; 'Up to 32 GB ECC...' -> 32."""
    if not isinstance(text, str):
        return None
    m = re.search(r"(\d[\d,]*)\s*(GB|MB)", text, re.I)
    if not m:
        return None
    n = float(m.group(1).replace(",", ""))
    return n / 1024 if m.group(2).upper() == "MB" else n


def build_wide(spec_long: pd.DataFrame) -> pd.DataFrame:
    wide = (spec_long.pivot_table(index=["product", "section", "page"], columns="attribute",
                                  values="value", aggfunc="first")
            .reset_index())
    wide.columns.name = None
    wide = wide.rename(columns={"page": "spec_page", "section": "spec_section"})
    # keep the natural attribute order from the catalogue, not alphabetical
    order = list(dict.fromkeys(spec_long["attribute"]))
    fixed = ["product", "spec_section", "spec_page"]
    return wide[fixed + [c for c in order if c in wide.columns]]


def normalise(wide: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"product": wide["product"]})
    get = lambda col: wide[col] if col in wide.columns else pd.Series([None] * len(wide))
    out["weight_kg"] = get("Weight").map(lambda t: first_number(t, "kg"))
    disp = get("Display size (diagonal)")
    out["display_cm"] = disp.map(lambda t: first_number(t, "cm"))
    out["display_inch"] = out["display_cm"].map(lambda c: round(c / 2.54, 1) if c else None)
    mem = get("Memory upgrade")
    out["max_memory_gb"] = mem.map(max_memory_gb)
    out["max_memory_gb"] = out["max_memory_gb"].fillna(get("Memory").map(max_memory_gb))
    return out


def style_workbook(path):
    wb = load_workbook(path)
    head_fill = PatternFill("solid", fgColor="0096D6")  # HP blue
    for ws in wb.worksheets:
        ws.freeze_panes = "A2"
        for c in ws[1]:
            c.font = Font(bold=True, color="FFFFFF")
            c.fill = head_fill
            c.alignment = Alignment(vertical="center", wrap_text=True)
        for col in ws.columns:
            width = max((len(str(c.value)) for c in col[:200] if c.value is not None), default=8)
            ws.column_dimensions[get_column_letter(col[0].column)].width = min(max(width + 2, 10), 70)
        for row in ws.iter_rows(min_row=2):
            for c in row:
                c.alignment = Alignment(vertical="top", wrap_text=True)
        ws.auto_filter.ref = ws.dimensions
    wb.save(path)


def main():
    rd = lambda n: pd.read_csv(OUT / n, keep_default_na=False)
    products = rd("products.csv")
    spec_long = rd("spec_tables_long.csv")
    care = rd("care_packs_long.csv")
    rec = rd("care_pack_recommended.csv")
    extras = rd("features_and_captions.csv")
    toc = rd("toc.csv")

    care["product_family"] = care["product_family"].map(balance_parens)
    care["section"] = care["section"].str.replace(r"\s*HP Care Pack\s*$", "", regex=True)
    care.to_csv(OUT / "care_packs_long.csv", index=False)

    wide = build_wide(spec_long)
    wide.to_csv(OUT / "spec_tables_wide.csv", index=False)
    norm = normalise(wide)

    # ---- join descriptions with specs on a normalised product name
    products = products.rename(columns={"page": "description_page", "section": "category"})
    products["_k"] = products["name"].map(norm_name)
    wide["_k"] = wide["product"].map(norm_name)
    norm["_k"] = wide["_k"]
    master = (products.drop(columns=["col_x", "row_y"])
              .merge(norm.drop(columns=["product"]), on="_k", how="left")
              .merge(wide.drop(columns=["product"]), on="_k", how="left")
              .drop(columns=["_k"]))
    master["has_spec_table"] = master["spec_page"].notna()
    master.to_csv(OUT / "products_master.csv", index=False)

    xlsx = OUT / "catalogue_extracted.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
        master.to_excel(xw, sheet_name="Products_Master", index=False)
        wide.drop(columns=["_k"]).to_excel(xw, sheet_name="Specs_Wide", index=False)
        spec_long.to_excel(xw, sheet_name="Specs_Long", index=False)
        care.to_excel(xw, sheet_name="Care_Packs", index=False)
        rec.to_excel(xw, sheet_name="Care_Pack_Recommended", index=False)
        extras.drop(columns=["col_x", "row_y"]).to_excel(xw, sheet_name="Features_Captions", index=False)
        toc.to_excel(xw, sheet_name="TOC", index=False)
    style_workbook(xlsx)
    print(f"master={len(master)} (with specs: {int(master.has_spec_table.sum())}) "
          f"wide={len(wide)} care={len(care)} -> {xlsx.name}")


if __name__ == "__main__":
    main()
