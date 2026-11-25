"""Streamlit chat UI for the HP catalogue RAG bot.

Run from the project root:
    streamlit run frontend/streamlit_app.py
Backend URL: environment variable BACKEND_URL (default http://localhost:8000).
"""
import json
import os

import httpx
import streamlit as st

BACKEND = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")

EXAMPLES = [
    "What is the Care Pack SKU for a 3-year next business day service for the ProBook 450 G2?",
    "Compare the HP EliteBook 840 G1 and EliteBook 850 G1.",
    "Which HP notebooks weigh less than 1.5 kg?",
    "What ports does the HP ProBook 640 G1 have?",
    "Which workstation supports the most memory?",
    "What displays does HP offer and what are their resolutions?",
]

st.set_page_config(page_title="HP Catalogue Assistant", page_icon="💻", layout="wide")


# ----------------------------------------------------------------------------- API helpers
@st.cache_data(ttl=15, show_spinner=False)
def api_get(path: str):
    try:
        r = httpx.get(f"{BACKEND}{path}", timeout=5)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:  # backend down, wrong URL, ...
        return None, f"{type(e).__name__}: {e}"


def stream_chat(payload: dict, holder: dict):
    """Generator of answer text; fills `holder` with sources / errors as events arrive."""
    with httpx.stream("POST", f"{BACKEND}/chat/stream", json=payload,
                      timeout=httpx.Timeout(180, connect=5)) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            ev = json.loads(line)
            if ev["type"] == "sources":
                holder["sources"] = ev["sources"]
                holder["llm"] = ev.get("llm", "")
            elif ev["type"] == "token":
                yield ev["text"]
            elif ev["type"] == "error":
                holder["error"] = ev["message"]


def show_sources(sources: list[dict]):
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            st.markdown(f"**[{s['n']}] {s['title']}** · {s['category']} · page {s['page']} "
                        f"· _{s['doc_type']}, matched by {s['matched_by']}_")
            st.caption(s["snippet"])


# ----------------------------------------------------------------------------- sidebar
health, health_err = api_get("/health")
cats, _ = api_get("/catalogue/categories")

with st.sidebar:
    st.header("HP Catalogue Assistant")
    if health_err:
        st.error(f"Backend not reachable at {BACKEND}\n\n{health_err}")
    else:
        if health["status"] != "ok":
            st.warning("Backend is up but has no data. Run `python scripts/run_pipeline.py`.")
        st.caption(f"Model: `{health['llm']}`")
        st.caption(f"Search: `{health['embedder']}` + keywords · "
                   f"{health['index'].get('n_chunks', 0)} chunks")

    st.subheader("Filters")
    options = [c for c in (cats or {}).get("categories", {}) if c != "Catalogue"]
    chosen = st.multiselect("Catalogue section", options, placeholder="All sections")
    top_k = st.slider("Passages to retrieve", 2, 12, 6)
    stream_on = st.toggle("Stream answer", value=True)
    show_src = st.toggle("Show sources", value=True)

    c1, c2 = st.columns(2)
    if c1.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    if c2.button("Rebuild index", use_container_width=True):
        try:
            httpx.post(f"{BACKEND}/index/rebuild", timeout=300).raise_for_status()
            st.cache_data.clear()
            st.success("Index rebuilt")
        except Exception as e:
            st.error(f"Rebuild failed: {e}")

# ----------------------------------------------------------------------------- chat
st.title("Ask the HP Catalogue 2014")
st.caption("Answers come only from the extracted catalogue (specs, descriptions, Care Pack SKUs). "
           "The catalogue has no prices.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = None

if not st.session_state.messages:
    st.write("Try one of these:")
    cols = st.columns(2)
    for i, q in enumerate(EXAMPLES):
        if cols[i % 2].button(q, key=f"ex{i}", use_container_width=True):
            st.session_state.pending = q
            st.rerun()

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant" and show_src:
            show_sources(m.get("sources", []))

question = st.chat_input("Ask about specs, models, comparisons or Care Pack SKUs…")
if st.session_state.pending:
    question, st.session_state.pending = st.session_state.pending, None

if question:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    payload = {"question": question, "history": history[-6:], "top_k": top_k,
               "categories": chosen or None}
    holder: dict = {}
    with st.chat_message("assistant"):
        try:
            if stream_on:
                answer = st.write_stream(stream_chat(payload, holder))
            else:
                with st.spinner("Searching the catalogue…"):
                    r = httpx.post(f"{BACKEND}/chat", json=payload, timeout=180)
                    r.raise_for_status()
                    data = r.json()
                answer, holder["sources"] = data["answer"], data["sources"]
                st.markdown(answer)
            if holder.get("error"):
                st.error(f"The language model failed: {holder['error']}")
                answer = (answer or "") + f"\n\n_(error: {holder['error']})_"
            if show_src:
                show_sources(holder.get("sources", []))
        except Exception as e:
            answer = f"Could not reach the backend at {BACKEND}: {e}"
            st.error(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer or "",
                                      "sources": holder.get("sources", [])})
