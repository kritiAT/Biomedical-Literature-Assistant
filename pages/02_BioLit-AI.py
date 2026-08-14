"""
streamlit_app.py -- Multimodal Biomedical RAG Assistant UI (MVP)

UI only -- all extraction/chunking/summarization/retrieval logic lives in rag_backend.py.

Run:
    # .env file with OPENAI_API_KEY=sk-...
    streamlit run streamlit_app.py
"""

import os
import re
import time
import tempfile

import streamlit as st

import rag_backend as rb

st.set_page_config(page_title="BioLit-AI", page_icon="🤖", layout="wide")

if not rb.OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found. Add it to a .env file next to this app and restart.")
    st.stop()

# --------------------------------------------------------------------------
# One-time layout model preload (cached across reruns/sessions in this server process)
# --------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading PDF layout model (first run only, ~30-60s)...")
def _preload_layout_model():
    try:
        return rb.load_layout_model()
    except Exception as e:
        st.warning(f"Could not preload layout model ({e}); it will load lazily on first parse instead.")
        return None


_preload_layout_model()

# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------

if "documents" not in st.session_state:
    st.session_state.documents = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "retriever" not in st.session_state:
    st.session_state.retriever = rb.init_retriever()

# --------------------------------------------------------------------------
# Sidebar: source mode, upload/PMCID input, document list
# --------------------------------------------------------------------------

st.markdown("""
<style>
h1 {
    background: linear-gradient(90deg, #6C63FF, #FF6584);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #6C63FF20, #FFFFFF);
}
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
    background-color: #EFF3FF;
}
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) {
    background-color: #FFF4EC;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Add articles")
    mode = st.radio("Source", ["Upload PDF", "Enter PMCIDs"], horizontal=True, label_visibility='collapsed')
    if mode == "Upload PDF":
        st.caption("🐢 Parses text, tables & images — thorough but slower.")
    else:
        st.caption("⚡ Text only (incl. table/figure captions) — fast.")

    uploaded_files = None
    pmcid_text = ""
    if mode == "Upload PDF":
        uploaded_files = st.file_uploader(
            f"Upload up to {rb.MAX_FILES} PDF research articles", type=["pdf"], accept_multiple_files=True,
        )
        if uploaded_files and len(uploaded_files) > rb.MAX_FILES:
            st.error(f"Please upload at most {rb.MAX_FILES} files (you selected {len(uploaded_files)}).")
            uploaded_files = uploaded_files[:rb.MAX_FILES]
    else:
        pmcid_text = st.text_area(
            f"Enter up to {rb.MAX_PMCIDS} PMCIDs (comma or newline separated)",
            placeholder="PMC7096066\nPMC8123456, PMC9012345",
        )

    process_clicked = st.button("Process documents", type="primary")

    if process_clicked:
        llm_summarize = rb.get_summarize_llm()
        already_ingested = {d["title"] for d in st.session_state.documents}

        with st.status("Processing documents...", expanded=True) as status:
            extraction_results = []

            if mode == "Upload PDF" and uploaded_files:
                for f in uploaded_files:
                    if f.name in already_ingested:
                        continue
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(f.read())
                        tmp_path = tmp.name
                    result = rb.extract_from_pdf(tmp_path, f.name, status_cb=lambda m: status.write(m))
                    extraction_results.append(result)
                    os.unlink(tmp_path)

            elif mode == "Enter PMCIDs" and pmcid_text.strip():
                raw_ids = [x.strip() for x in re.split(r"[,\n]", pmcid_text) if x.strip()]
                if len(raw_ids) > rb.MAX_PMCIDS:
                    status.write(f"Only the first {rb.MAX_PMCIDS} PMCIDs will be processed.")
                for raw_id in raw_ids[:rb.MAX_PMCIDS]:
                    result = rb.extract_from_pmcid(raw_id, status_cb=lambda m: status.write(m))
                    extraction_results.append(result)
                    time.sleep(0.35)  # stay under NCBI's unauthenticated rate limit

            for result in extraction_results:
                label = result["source_ref"]
                if result["status"] != "success":
                    status.write(f"FAILED {label}: {result['message']}")
                    continue

                indexed = rb.summarize_and_index(result, st.session_state.retriever, llm_summarize,
                                                    status_cb=lambda m: status.write(m))
                st.session_state.documents.append(indexed)
                status.write(f"DONE {indexed['title']}: indexed ({indexed['num_text_chunks']} text, "
                              f"{indexed['num_tables']} tables, {indexed['num_images']} images).")

            status.update(label="Done processing documents.", state="complete")

    st.divider()
    st.header("Documents in index")
    if not st.session_state.documents:
        st.caption("No documents processed yet.")
    else:
        selected_ids = []
        for doc in st.session_state.documents:
            label = f"{doc['title']}  ({doc['num_text_chunks']} text, {doc['num_tables']} tables, {doc['num_images']} images)"
            if st.checkbox(label, value=True, key=f"doc_{doc['document_id']}"):
                selected_ids.append(doc["document_id"])
        st.session_state.selected_document_ids = selected_ids

# --------------------------------------------------------------------------
# Main: chat interface
# --------------------------------------------------------------------------

st.title("🤖 BioLit-AI — Biomedical Literature Assistant")
st.caption(
"Ask natural-language questions across your uploaded research articles and get grounded answers with citations. "
"Add papers as PDFs or by PMCID in the sidebar to get started."
)

def render_sources(sources):
    with st.expander("Sources"):
        for s in sources:
            icon = {"text": "📄", "table": "📊", "image": "🖼️"}.get(s["content_type"], "-")
            section_str = s["section"] + (f" > {s['subsection']}" if s.get("subsection") else "")
            st.markdown(f"**{icon} {s['title']}** — {section_str}" + (f", p.{s['page']}" if s.get("page") else ""))
            if s["content_type"] == "image":
                st.image(s["content"], width=300)
            elif s["content_type"] == "table":
                st.markdown(s["content"], unsafe_allow_html=True)
            else:
                st.text(s["content"][:500] + ("..." if len(s["content"]) > 500 else ""))


for turn in st.session_state.chat_history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("sources"):
            render_sources(turn["sources"])

query = st.chat_input("Ask a question about the uploaded articles...")

if query:
    if not st.session_state.documents:
        st.error("Upload and process at least one article first.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving and synthesizing answer..."):
                llm_answer = rb.get_answer_llm()
                doc_ids = st.session_state.get("selected_document_ids") or None
                answer, sources = rb.answer_question(
                    query, st.session_state.retriever, llm_answer, k=5, document_ids=doc_ids
                )
                st.markdown(answer)
                if sources:
                    render_sources(sources)

        st.session_state.chat_history.append({"role": "assistant", "content": answer, "sources": sources})
