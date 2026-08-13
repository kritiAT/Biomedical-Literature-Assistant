"""
Melanoma Research Assistant — Homepage

A simple landing page introducing the app. Intended as the entry point of a
Streamlit multipage app (e.g. save this as `Home.py` at the project root, with
the chat assistant and future PDF/RAG page saved under a `pages/` folder —
Streamlit auto-detects and lists them in the sidebar).

Run locally:
    streamlit run Home.py
"""

import streamlit as st

# ----------------------------------------------------------------------------
# Static knowledge-base metrics — update manually after each ingestion run
# (see notebooks 1 and 3). Kept static and in one place so both this homepage
# and the chat page can be updated consistently.
# ----------------------------------------------------------------------------
NUM_ARTICLES_INDEXED = 1850   # <-- set to your actual ingested PubMed article count
NUM_DRUGS_COVERED = 10        # <-- keep in sync with MELANOMA_DRUGS in the other pages

st.set_page_config(page_title="Melanoma Research Assistant", page_icon="🔬", layout="centered")

st.title("🔬 Melanoma Research Assistant")
st.caption("An AI research companion for melanoma literature and drug information.")

st.divider()

# --- Knowledge base metrics ---
m1, m2 = st.columns(2)
m1.metric("📚 PubMed articles indexed", f"{NUM_ARTICLES_INDEXED:,}")
m2.metric("💊 Drugs covered", f"{NUM_DRUGS_COVERED}")

st.divider()

# --- What this app does ---
st.subheader("About this assistant")
st.markdown(
    """
This assistant helps clinicians and researchers explore melanoma-related knowledge quickly,
with every answer grounded in real sources rather than the model's general knowledge.

- **Literature search** — retrieves relevant abstracts from PubMed and summarizes them with
  inline citations (PMID).
- **Drug information** — looks up melanoma-relevant drugs using structured data from **PubChem** and **ChEMBL**, including mechanism of
  action and indications.
- **Conversational memory** — remembers the current conversation.
- **Scope** — focused on melanoma only.

⚠️ This tool summarizes published research and drug data for informational purposes.
It is **not medical advice** and does not replace clinical judgment.
"""
)

st.divider()

# --- Coming soon: PDF upload + RAG ---
st.subheader("📄 Upload your own papers")
st.markdown(
    """
A second assitant lets you **upload your own PDFs** — such as a specific
research paper or clinical trial report — and ask questions about it
directly.

- Upload one or more PDFs related to biomedical research.
- The assistant will chunk, embed, and index the PDF content in a dedicated retrieval pipeline
  (separate from the main PubMed/drug knowledge base).
- Ask questions and get answers grounded specifically in **your uploaded documents**, with
  citations back to the page/section they came from.

This will be useful for reviewing a new paper, cross-checking a document against the existing
melanoma literature, or asking questions about material not yet in the indexed knowledge base.
"""
)

st.divider()

st.info("👈 Use the sidebar to navigate to the chat assistant or the PDF upload page.")