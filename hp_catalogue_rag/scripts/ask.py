"""Ask a question from the terminal (no servers needed):

    python scripts/ask.py "What is the SKU for a 3-year NBD care pack for the ProBook 450 G2?"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.config import get_settings  # noqa: E402
from backend.app.rag import RAGService  # noqa: E402

if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Which HP notebooks are the lightest?"
    r = RAGService(get_settings()).answer(q)
    print(r["answer"], "\n\nSources:")
    for s in r["sources"]:
        print(f"  [{s['n']}] {s['title']} - page {s['page']} ({s['doc_type']})")
