"""One command for the whole data pipeline:

    PDF -> split spreads -> extract -> normalise/merge -> validate -> RAG index

    python scripts/run_pipeline.py                     # everything
    python scripts/run_pipeline.py --skip-extraction   # only rebuild the search index
    python scripts/run_pipeline.py --with-unstructured # also run the Unstructured.io hi_res route
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

STEPS = [
    ("1/5 split two-page spreads", "s1_split_spreads.py"),
    ("2/5 extract products, specs, Care Pack SKUs", "s3_native_extract.py"),
    ("3/5 normalise + merge + Excel", "s4_build_outputs.py"),
    ("4/5 validate against the PDF", "s5_validate.py"),
]


def run(label: str, script: str, extra: list[str] | None = None):
    print(f"\n=== {label} ===", flush=True)
    t = time.time()
    r = subprocess.run([sys.executable, "-W", "ignore", str(ROOT / "extraction" / script),
                        *(extra or [])], cwd=ROOT)
    if r.returncode:
        sys.exit(f"\nStep failed: {script} (exit {r.returncode}). "
                 f"{'See output/qa_report.md for details.' if script.startswith('s5') else ''}")
    print(f"    done in {time.time() - t:.1f}s")


def build_index():
    print("\n=== 5/5 build RAG index ===", flush=True)
    from backend.app.config import get_settings
    from backend.app.embeddings import get_embedder
    from backend.app.index import HybridIndex

    s = get_settings()
    idx = HybridIndex.build(s.data_dir, get_embedder(s))
    idx.save(s.index_dir)
    print(f"    {idx.meta['n_chunks']} chunks, embedder={idx.meta['embedder']} -> {s.index_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-extraction", action="store_true")
    ap.add_argument("--with-unstructured", action="store_true",
                    help="also run extraction/s2_unstructured_hi_res.py (needs poppler, tesseract, internet)")
    a = ap.parse_args()

    if not a.skip_extraction:
        if not (ROOT / "data" / "input" / "catalogue.pdf").exists():
            sys.exit("Put the catalogue at data/input/catalogue.pdf first.")
        for label, script in STEPS:
            run(label, script)
        if a.with_unstructured:
            run("extra: Unstructured.io hi_res", "s2_unstructured_hi_res.py")
    build_index()
    print("\nAll done. Start the app:  python scripts/dev_server.py")


if __name__ == "__main__":
    main()
