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
NUM_ARTICLES_INDEXED = 4783
NUM_DRUGS_COVERED = 57

st.set_page_config(page_title="Melanoma Research Assistant", page_icon="🧠", layout="wide")

st.markdown("""
<style>
h1 {
    background: linear-gradient(90deg, #11998e, #38ef7d);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #11998e20, #FFFFFF);
}
</style>
""", unsafe_allow_html=True)

st.title("🔬 Melanoma Research Assistant")
st.caption("An AI research companion for melanoma literature and drug information.")

st.divider()

# --- Knowledge base metrics ---
m1, m2 = st.columns(2)
m1.markdown(
            f'<div style="background-color:#EFF3FF; padding:10px 14px; '
            f'border-radius:12px; line-height:1.5;">📚 <b>{NUM_ARTICLES_INDEXED:,}</b> PubMed Articles Indexed</div>',
            unsafe_allow_html=True,
        )
m2.markdown(
            f'<div style="background-color:#E8F5E9; padding:10px 14px; '
            f'border-radius:12px; line-height:1.5;">💊 <b>{NUM_DRUGS_COVERED}</b> Drugs Covered</div>',
            unsafe_allow_html=True,
        )

st.divider()

# --- What this app does ---
st.subheader("🔍 Mel-AI — Melanoma Research Assistant")
st.markdown("""
Mel-AI helps clinicians and researchers explore melanoma knowledge quickly and reliably. 

💬 Ask a question, and it searches curated 📚 PubMed literature and 💊 PubChem/ChEMBL drug data, then generates a grounded, cited answer — not a guess from general knowledge. 

🧠 It remembers your conversation, so follow-up questions come naturally, and it stays focused on melanoma alone to keep every answer relevant and precise.

⚠️ Built for exploration and literature review — not a substitute for clinical judgment or medical advice.""")

st.divider()

# --- Coming soon: PDF upload + RAG ---
st.subheader("🤖 BioLit-AI — Biomedical Literature Assistant")
st.markdown("""
Upload research papers (PDF) or fetch them straight from PubMed Central by PMCID, then ask questions in plain English. 

🔍 BioLit-AI reads text, tables, and figures across up to 10 articles at once and answers with citations — section, subsection, and page or PMC link — so every claim traces back to the source. 

⚡ Fast text-only mode for PMCIDs, full multimodal parsing for PDFs.

📄 Useful for reviewing a new paper, cross-checking a document against the existing melanoma literature, or asking questions about material not yet in the indexed knowledge base.
""")

st.info("👈 Use the sidebar to navigate to the chat assistants.")