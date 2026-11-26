# HP Catalogue RAG Chatbot

Extract data from the **HP PC Catalogue 2014** PDF and chat with it.
Backend: **FastAPI** · Frontend: **Streamlit** · LLM: **NVIDIA API**. Works with **no API key** too (answers become catalogue excerpts).

---

## 1. The idea in 60 seconds

Imagine a librarian who has to answer questions about a 60-page catalogue.

1. **Reading** – she first copies every product, spec table and warranty SKU into neat index cards (this is the *extraction*).
2. **Finding** – when you ask something, she does not re-read the whole book. She pulls the 6 most relevant cards (this is *retrieval*).
3. **Answering** – she writes the answer using only those cards and tells you which card each fact came from (this is *generation* by an LLM, with citations).

```
catalogue.pdf
   │  extraction/  split spreads → read tables by their drawn lines → clean → validate
   ▼
output/*.csv  (products, specs, Care Pack SKUs, ...)          ← the "index cards" source
   │  backend/app/documents.py   one chunk per product / spec table / SKU block
   ▼
data/index/  (BM25 keywords  [+ semantic vectors])
   │
   │      Streamlit  ──question──▶  FastAPI  /chat/stream
   │      (frontend)               1. retrieve top chunks (hybrid search + product-name lookup)
   │          ▲                    2. build a grounded prompt
   └─answer + sources◀──────────── 3. LLM writes the answer (or extractive fallback)
```

## 2. Quick start

Python **3.10+** required.

```bash
# 1. install
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. (optional) choose your LLM: copy .env.example to .env and paste ONE key
cp .env.example .env                 # Windows: copy .env.example .env
#    NVIDIA:  NVIDIA_API_KEY=nvapi-...     (key from build.nvidia.com)

# 3. (optional) rebuild the data from the PDF – the extracted data is already in output/
python scripts/run_pipeline.py

# 4. start backend + frontend together
python scripts/dev_server.py
```

Open **http://localhost:8501** (chat) and **http://localhost:8000/docs** (API docs).
The first start builds the search index automatically (a second or two).

Without step 2 the app still works: answers are the best-matching catalogue excerpts.
Quick terminal test without any server: `python scripts/ask.py "Which HP notebooks are the lightest?"`

Run separately instead of `dev_server.py`:
```bash
uvicorn backend.app.main:app --reload --port 8000
BACKEND_URL=http://localhost:8000 streamlit run frontend/streamlit_app.py     # Windows: set BACKEND_URL=... first
```

## 3. Project layout

```
hp_catalogue_rag/
├── data/input/catalogue.pdf          the source PDF
├── extraction/                       PDF → structured data
│   ├── s1_split_spreads.py           cut two-page spreads into single pages (60 printed pages)
│   ├── s2_unstructured_hi_res.py     OPTIONAL Unstructured.io hi_res route (typed elements, HTML tables, chunks)
│   ├── s3_native_extract.py          main extractor: tables, products, Care Pack SKUs
│   ├── s4_build_outputs.py           merge, normalise (kg, inch, GB), Excel workbook
│   ├── s5_validate.py                independent checks → output/qa_report.md
│   └── common.py
├── output/                           extracted data (ready to use)
├── backend/app/                      FastAPI RAG service
│   ├── documents.py                  data → chunks
│   ├── index.py                      BM25 + optional dense + product-name lookup
│   ├── embeddings.py                 sentence-transformers / OpenAI (optional)
│   ├── llm.py                        Anthropic / OpenAI / Ollama / none
│   ├── rag.py                        retrieve → prompt → answer + sources
│   ├── config.py                     settings (env / .env)
│   └── main.py                       HTTP API
├── frontend/streamlit_app.py         chat UI
├── scripts/                          run_pipeline.py · dev_server.py · ask.py
├── tests/                            20 tests (pytest -q): retrieval, API, NVIDIA wire format
├── requirements.txt · requirements-optional.txt · .env.example
```

## 4. What gets extracted (`output/`)

| File | What it holds |
|---|---|
| `products.csv` / `.json` | 126 products: name, catalogue section, page, description |
| `spec_tables_long.csv` | 670 rows: product × attribute × value (every spec table) |
| `spec_tables_wide.csv` | one row per product, one column per attribute |
| `products_master.csv` | descriptions + specs joined, plus `weight_kg`, `display_inch`, `max_memory_gb` |
| `care_packs_long.csv` | 377 Care Pack SKUs: family × service level × term → SKU (+ base warranty) |
| `care_pack_recommended.csv` | recommended service descriptions |
| `features_and_captions.csv` · `pages.jsonl` · `toc.csv` | feature blurbs, per-page text/footnotes, table of contents |
| `catalogue_extracted.xlsx` | everything above as a formatted workbook |
| `tables_raw.json` | raw cell grid per table with bounding boxes (for debugging) |
| `qa_report.md` | validation: SKU recount per page + table text coverage |

**Why not only Unstructured `hi_res`?** The catalogue is a *digital* PDF, so every table is drawn with real lines and every character carries its font and position. `s3_native_extract.py` reads that structure directly: it is faster, deterministic and was verified (all 377 SKUs match a plain-text recount; table coverage 95–100%). `s2_unstructured_hi_res.py` stays available as the layout-model route (`python scripts/run_pipeline.py --with-unstructured`; needs poppler + tesseract and internet on first run to download the models). Use it as a second opinion or when you adapt the pipeline to a *scanned* PDF.

## 5. How the chatbot finds the right cards

* **Chunks = meaningful units**, not 500-character slices: one overview and one spec table per product, one block per Care Pack family × service level (each lists the models it applies to: "ProBook 4xx → 430, 450, 455, 470 G2"), recommended services, feature blurbs, footnotes, and one *comparison table per category* for "which has the most / lightest…" questions.
* **Hybrid search**: BM25 keywords (great for SKUs like `UK703E`, model names like `450 G2`) + optional semantic vectors, merged with Reciprocal Rank Fusion.
* **Product-name lookup**: if you mention "ProBook 450 G2", its overview and specs are always included.
* **Follow-ups**: a short question ("and the 5-year one?") borrows words from your previous question.
* **Grounded prompt**: the LLM must answer only from numbered passages and cite `[1] [2]`; the UI shows the sources with page numbers.

## 6. Configuration (`.env`)

| Variable | Default | Meaning |
|---|---|---|
| `NVIDIA_API_KEY` | – | your `nvapi-...` key from build.nvidia.com |
| `NVIDIA_MODEL` | `meta/llama-3.3-70b-instruct` | any chat model listed on build.nvidia.com |
| `NVIDIA_BASE_URL` | `https://integrate.api.nvidia.com/v1` | NVIDIA OpenAI-compatible API endpoint |
| `NVIDIA_EMBEDDING_MODEL` | `nvidia/nv-embedqa-e5-v5` | used when `EMBEDDING_PROVIDER=nvidia` |
| `EMBEDDING_PROVIDER` | `nvidia` | `nvidia`, `bm25`, or `sentence-transformers` |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | downloaded on first use (~130 MB) |
| `TOP_K` | `6` | passages sent to the LLM |
| `BACKEND_URL` | `http://localhost:8000` | used by Streamlit |

**Using only an NVIDIA key (recommended setup):**
```
NVIDIA_API_KEY=nvapi-xxxxxxxx
EMBEDDING_PROVIDER=nvidia      # default; semantic search through NVIDIA too
```
Restart the backend. The sidebar should show `nvidia:meta/llama-3.3-70b-instruct`. (Free NVIDIA keys have rate limits; if you see HTTP 429 errors wait a moment or lower `TOP_K`.)

For local semantic search instead: `pip install sentence-transformers` (pulls in PyTorch), restart, done. The index rebuilds itself when the data or the embedding model changes.

## 7. API

| Endpoint | Purpose |
|---|---|
| `GET /health` | status, LLM, embedder, chunk count |
| `POST /chat` | `{question, history?, top_k?, categories?, doc_types?}` → `{answer, sources, llm}` |
| `POST /chat/stream` | same input; newline-delimited JSON events `sources` → `token`* → `done` (or `error`) |
| `POST /search` | retrieval only (no LLM) – to debug what the bot can see |
| `GET /catalogue/categories` | sections and chunk types for filters |
| `POST /index/rebuild` | re-read `output/` and rebuild the index |

```bash
curl -s localhost:8000/chat -H 'content-type: application/json' \
  -d '{"question":"SKU for a 3-year next business day care pack for the ProBook 450 G2?"}'
```

## 8. Tests

```bash
pytest -q
```
Checks that every extracted SKU is searchable, every spec product is found by name, the ProBook 450 G2 care-pack question lands on the right table (`UK703E`), aggregate questions get comparison tables, the dense/hybrid path, and the API (including streaming and LLM-error handling) with a fake LLM, plus the NVIDIA provider against a local fake of its endpoints.

## 9. Using a different PDF

`s3_native_extract.py` relies on this catalogue's layout (spreads, ruled tables, bold headings, blue section headers). For another PDF, reuse `s1` + `s2` (layout-model route works on any PDF), or adapt `s3`. Everything downstream (`backend/`, `frontend/`) only needs the CSVs in `output/`, so it keeps working as long as you produce the same columns.

## 10. Known limitations

* The catalogue contains **no prices**; the bot says so instead of guessing.
* Care Pack tables for **monitors** are classed by screen size, so they cannot be linked to individual models.
* Questions that need maths across many models are answered from the comparison tables (abridged text); an LLM does the comparing, so check numbers against the cited page.
* Answers are only as good as retrieval: use **Sources** in the UI (or `/search`) to see what the model was given.

## 11. Troubleshooting

| Symptom | Fix |
|---|---|
| UI says "Backend not reachable" | start the backend (`python scripts/dev_server.py`) or set `BACKEND_URL` |
| "No LLM key set" banner | add `NVIDIA_API_KEY` to `.env` (project root) and restart the backend |
| `401` / `403` from NVIDIA | key must start with `nvapi-`; regenerate it on build.nvidia.com |
| `404` model not found | pick a model name exactly as shown on build.nvidia.com and set `NVIDIA_MODEL` |
| `410` from NVIDIA during startup | the embedding service/model may be unavailable; set `EMBEDDING_PROVIDER=bm25` to start with keyword search, or choose an available embedding model |
| `no data` status | run `python scripts/run_pipeline.py` (or check `DATA_DIR`) |
| Semantic model download fails | the app falls back to keyword search; set `EMBEDDING_PROVIDER=bm25` to silence it |
| Changed `output/` by hand | click **Rebuild index** in the sidebar or `POST /index/rebuild` |
