"""Streamlit UI: upload PDFs, index them, and chat with citations.

Run with ``streamlit run app.py``.
"""

from __future__ import annotations

import streamlit as st

from documind.config import setup_logging
from documind.rag import DocuMind
from documind.retrieve import MODES

setup_logging()
st.set_page_config(page_title="DocuMind", page_icon="📄", layout="wide")


@st.cache_resource
def get_engine() -> DocuMind:
    return DocuMind()


engine = get_engine()

with st.sidebar:
    st.header("Documents")
    uploads = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    if st.button("Ingest", type="primary", use_container_width=True):
        engine.settings.docs_dir.mkdir(parents=True, exist_ok=True)
        for f in uploads or []:
            (engine.settings.docs_dir / f.name).write_bytes(f.getvalue())
        with st.spinner("Chunking and embedding..."):
            report = engine.ingest()
        st.success(
            f"Added {len(report.added)}, updated {len(report.updated)}, "
            f"skipped {len(report.skipped)} unchanged ({report.chunks_written} new chunks)."
        )
    pdfs = sorted(p.name for p in engine.settings.docs_dir.glob("*.pdf"))
    st.caption(f"{len(pdfs)} PDFs in folder · {engine.store.count()} chunks indexed")
    for name in pdfs:
        st.markdown(f"- `{name}`")

    st.header("Retrieval")
    mode = st.selectbox("Mode", MODES, index=MODES.index(engine.settings.retrieval_mode))
    k = st.slider("Top-k chunks", 1, 10, engine.settings.top_k)
    st.caption(f"Model: `{engine.settings.model}` · effort `{engine.settings.effort}`")

st.title("📄 DocuMind")
st.write("Ask questions about your technical PDFs. Every answer cites the file and page it came from.")

if "history" not in st.session_state:
    st.session_state.history = []


def render(entry: dict) -> None:
    with st.chat_message("user"):
        st.write(entry["question"])
    with st.chat_message("assistant"):
        st.markdown(entry["answer"])
        st.caption(
            f"⏱ {entry['latency_ms']:.0f} ms · 💲{entry['cost_usd']:.4f} · "
            f"{entry['usage']['input_tokens']} in / {entry['usage']['output_tokens']} out tokens · "
            f"{entry['model']}"
        )
        with st.expander(f"Sources ({len(entry['chunks'])} retrieved passages)"):
            cited = {(s["source"], s["page"]) for s in entry["sources"]}
            for c in entry["chunks"]:
                mark = " ✅ cited" if (c["source"], c["page"]) in cited else ""
                st.markdown(f"**#{c['rank']} · {c['source']} · page {c['page']}** (score {c['score']}){mark}")
                st.text(c["text"])


for entry in st.session_state.history:
    render(entry)

if question := st.chat_input("e.g. How many GPIO pins does the Raspberry Pi Pico expose?"):
    if engine.store.count() == 0:
        st.warning("No documents indexed yet. Upload PDFs and click Ingest.")
    else:
        with st.spinner("Retrieving and asking Claude..."):
            result = engine.ask(question, k=k, mode=mode).to_dict()
        st.session_state.history.append(result)
        render(result)
