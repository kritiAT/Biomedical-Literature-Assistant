"""
Melanoma Research Assistant — Streamlit UI (Literature + Drug Data + Memory)

Combines two Pinecone namespaces:
  - default namespace  -> PubMed literature chunks (from 01_data_ingestion_embedding.ipynb)
  - "drugs" namespace   -> PubChem/ChEMBL drug chunks (from 03_drug_extension_rag_memory.ipynb)

A lightweight LLM router decides, per question, whether to search literature, drugs, or both.
Conversation memory is kept in st.session_state and fed back into the prompt on every turn, so
follow-up questions ("what about combined with trametinib?") resolve correctly.

Run locally:
    streamlit run streamlit_app_drugs_memory.py

Required environment variables (.env file or host secrets):
    OPENAI_API_KEY
    PINECONE_API_KEY

Install dependencies:
    pip install streamlit langchain langchain-core langchain-text-splitters langchain-openai \\
                langchain-pinecone pinecone pydantic python-dotenv

NOTE: install "pinecone" (not the deprecated "pinecone-client") — see project notes on
package-name conflicts and LangChain 1.0's restructured import paths (langchain_text_splitters,
langchain_core.documents) if you hit install/import errors.
"""

import os
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
PINECONE_INDEX_NAME = "melanoma-research-assistant"
LITERATURE_NAMESPACE = ""      # default namespace, matches notebook 1
DRUGS_NAMESPACE = "drugs"      # matches notebook 3
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
TOP_K = 5
MAX_HISTORY_TURNS = 6          # cap conversation memory window sent to the LLM

# Static knowledge-base metrics shown in the UI.
NUM_ARTICLES_INDEXED = 2000
NUM_DRUGS_COVERED = 10

SUGGESTED_QUESTIONS = [
    "What does the literature say about survival rates for stage IV melanoma?",
    "What is the mechanism of action of vemurafenib?",
    "Compare the mechanism of dabrafenib with recent trial outcomes combining it with trametinib",
]

# Phrases that indicate the assistant couldn't really answer — used to suppress the
# "Sources" panel so empty/irrelevant citations aren't shown alongside a non-answer.
NON_ANSWER_PHRASES = [
    "cannot provide", "can't provide", "cannot answer", "can't answer",
    "do not have enough information", "don't have enough information",
    "not enough information", "does not contain enough information",
    "no relevant melanoma", "unable to answer", "outside the scope",
    "scoped to melanoma research only",
]

SYSTEM_PROMPT = """You are a friendly melanoma research assistant for clinicians and researchers.

Rules you must follow:
- Only answer questions related to melanoma (research literature, biology, and melanoma-relevant drugs).
- Base your answer ONLY on the provided context below plus the conversation history. Do not use outside knowledge.
- Cite literature claims as (PMID: xxxxx) and drug claims as (Drug: <name>, ChEMBL: <id>).
- If the context does not contain enough information, say so explicitly instead of guessing.
- This is a research/literature summary tool, not medical advice — do not give definitive treatment recommendations.

Context:
{context}
"""


class RouteDecision(BaseModel):
    query_literature: bool = Field(
        description="True if the question needs published melanoma research/clinical literature."
    )
    query_drugs: bool = Field(
        description="True if the question needs drug-specific data (mechanism, structure, targets, indications)."
    )


# ----------------------------------------------------------------------------
# Cached resources — built once per session. show_spinner text gives users
# feedback during the (slower) first load instead of a blank screen.
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="🔄 Connecting to the melanoma knowledge base...")
def load_resources():
    openai_api_key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    pinecone_api_key = os.environ.get("PINECONE_API_KEY") or st.secrets.get("PINECONE_API_KEY")

    if not openai_api_key or not pinecone_api_key:
        st.error(
            "Missing API keys. Set OPENAI_API_KEY and PINECONE_API_KEY as environment "
            "variables or in Streamlit secrets (.streamlit/secrets.toml)."
        )
        st.stop()

    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(PINECONE_INDEX_NAME)
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=openai_api_key)

    literature_vs = PineconeVectorStore(
        index=index, embedding=embeddings, text_key="text", namespace=LITERATURE_NAMESPACE
    )
    drugs_vs = PineconeVectorStore(
        index=index, embedding=embeddings, text_key="text", namespace=DRUGS_NAMESPACE
    )
    literature_retriever = literature_vs.as_retriever(search_kwargs={"k": TOP_K})
    drugs_retriever = drugs_vs.as_retriever(search_kwargs={"k": TOP_K})

    router_llm = ChatOpenAI(
        model=LLM_MODEL, temperature=0.1, openai_api_key=openai_api_key
    ).with_structured_output(RouteDecision)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0.2, openai_api_key=openai_api_key)
    chain = prompt | llm | StrOutputParser()

    return index, literature_retriever, drugs_retriever, router_llm, chain


def route_question(question: str, router_llm) -> RouteDecision:
    try:
        return router_llm.invoke(
            f"Classify what data sources this melanoma research question needs: '{question}'"
        )
    except Exception:
        # fail-safe: default to literature-only if the router call itself errors
        return RouteDecision(query_literature=True, query_drugs=False)


def format_doc(doc, source_type: str) -> str:
    meta = doc.metadata
    if source_type == "literature":
        header = (
            f"[LITERATURE | PMID: {meta.get('pmid')}] {meta.get('title')} "
            f"({meta.get('year')}, {meta.get('journal')})"
        )
    else:
        header = (
            f"[DRUG | {meta.get('drug_name', '').title()} | ChEMBL: {meta.get('chembl_id')} "
            f"| PubChem CID: {meta.get('pubchem_cid')}]"
        )
    return f"{header}\n{doc.page_content}"


def retrieve_combined(question: str, literature_retriever, drugs_retriever, router_llm):
    route = route_question(question, router_llm)
    if not route.query_literature and not route.query_drugs:
        route.query_literature = True  # fail-safe default

    lit_docs = literature_retriever.invoke(question) if route.query_literature else []
    drug_docs = drugs_retriever.invoke(question) if route.query_drugs else []

    context = "\n\n---\n\n".join(
        [format_doc(d, "literature") for d in lit_docs] + [format_doc(d, "drug") for d in drug_docs]
    )
    return context, lit_docs, drug_docs


def format_sources(lit_docs, drug_docs) -> list[dict]:
    sources, seen_pmids, seen_drugs = [], set(), set()
    for d in lit_docs:
        pmid = d.metadata.get("pmid")
        if pmid and pmid not in seen_pmids:
            seen_pmids.add(pmid)
            sources.append(
                {
                    "type": "literature",
                    "label": f"PMID {pmid}: {d.metadata.get('title')}",
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                }
            )
    for d in drug_docs:
        name = d.metadata.get("drug_name")
        if name and name not in seen_drugs:
            seen_drugs.add(name)
            cid = d.metadata.get("pubchem_cid")
            sources.append(
                {
                    "type": "drug",
                    "label": f"Drug: {name.title()} (ChEMBL {d.metadata.get('chembl_id')})",
                    "url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}" if cid else None,
                }
            )
    return sources


def is_non_answer(text: str) -> bool:
    """True if the answer indicates the assistant couldn't really help — used to
    suppress an irrelevant/empty Sources panel on refusals and out-of-scope replies."""
    lowered = text.lower()
    return any(phrase in lowered for phrase in NON_ANSWER_PHRASES)


def to_lc_history(turns: list[dict], max_turns: int = MAX_HISTORY_TURNS) -> list:
    """Convert Streamlit's stored dict messages into LC message objects, windowed to bound cost."""
    history = []
    for m in turns[-max_turns * 2:]:
        if m["role"] == "user":
            history.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            history.append(AIMessage(content=m["content"]))
    return history


def answer_question(question: str, conversation: list[dict], literature_retriever, drugs_retriever, router_llm, chain) -> dict:
    context, lit_docs, drug_docs = retrieve_combined(question, literature_retriever, drugs_retriever, router_llm)
    if not lit_docs and not drug_docs:
        return {
            "answer": "I'm sorry, I cannot provide more information — no relevant melanoma literature or drug data was found for this question.",
            "sources": [],
        }

    chat_history = to_lc_history(conversation)
    answer_text = chain.invoke({"context": context, "question": question, "chat_history": chat_history})

    # Suppress sources on a non-answer (out-of-scope question, or "not enough info" reply)
    # even if some docs were retrieved — showing citations next to "I don't know" is misleading.
    sources = [] if is_non_answer(answer_text) else format_sources(lit_docs, drug_docs)
    return {"answer": answer_text, "sources": sources}


def render_sources(sources: list[dict]) -> None:
    with st.expander("📎 Sources"):
        for s in sources:
            icon = "📄" if s["type"] == "literature" else "💊"
            if s["url"]:
                st.markdown(f"- {icon} [{s['label']}]({s['url']})")
            else:
                st.markdown(f"- {icon} {s['label']}")


def render_chat_message(role: str, content: str) -> None:
    """Render a chat bubble with role-specific background color (light theme)."""
    bubble_color = "#E3F2FD" if role == "user" else "#E8F5E9"  # light blue vs light green
    avatar = "🧑" if role == "user" else "🤖"
    with st.chat_message(role, avatar=avatar):
        st.markdown(
            f'<div style="background-color:{bubble_color}; padding:10px 14px; '
            f'border-radius:12px; line-height:1.5;">{content}</div>',
            unsafe_allow_html=True,
        )


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Melanoma Research Assistant", page_icon="🔬", layout="wide")

# Light theme touches: soft page background, rounded suggestion buttons, lighter sidebar.
st.markdown(
    """
    <style>
    .stApp { background-color: #FAFCFF; }
    section[data-testid="stSidebar"] { background-color: #F3F7FB; }
    div.stButton > button {
        border-radius: 20px;
        border: 2px solid #BBD6F0;
        background-color: #FFFFFF;
        color: #1B4965;
    }
    div.stButton > button:hover {
        border-color: #5FA8D3;
        color: #0B3954;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔬 Melanoma Research Assistant - MelAI")
st.caption(
    "Ask about melanoma research literature or melanoma-relevant drugs. Answers are grounded in "
    "retrieved PubMed abstracts and PubChem/ChEMBL drug data, with citations. Remembers this "
    "conversation's context. Not medical advice."
)

index, literature_retriever, drugs_retriever, router_llm, chain = load_resources()

# --- Knowledge base metrics row ---
m1, m2 = st.columns(2)
m1.metric("📚 PubMed articles indexed", f"{NUM_ARTICLES_INDEXED:,}")
m2.metric("💊 Drugs covered", f"{NUM_DRUGS_COVERED}")

st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# --- Suggested questions (shown only before the first message) ---
if not st.session_state.messages:
    st.markdown("**Try asking:**")
    cols = st.columns(3)
    for col, question in zip(cols, SUGGESTED_QUESTIONS):
        if col.button(question, use_container_width=True):
            st.session_state.pending_question = question

# --- Render chat history ---
for msg in st.session_state.messages:
    render_chat_message(msg["role"], msg["content"])
    if msg["role"] == "assistant" and msg.get("sources"):
        render_sources(msg["sources"])

# --- Input: typed question or a clicked suggestion ---
typed_question = st.chat_input("e.g. What about dabrafenib combined with trametinib?")
question_to_process = typed_question or st.session_state.pending_question
st.session_state.pending_question = None  # reset so it only fires once

if question_to_process:
    st.session_state.messages.append({"role": "user", "content": question_to_process})
    render_chat_message("user", question_to_process)

    with st.chat_message("assistant", avatar="🔬"):
        with st.spinner("Routing question, searching sources, and generating an answer..."):
            result = answer_question(
                question_to_process,
                st.session_state.messages[:-1],  # history excludes the question just asked
                literature_retriever,
                drugs_retriever,
                router_llm,
                chain,
            )
            st.markdown(
                f'<div style="background-color:#E8F5E9; padding:10px 14px; '
                f'border-radius:12px; line-height:1.5;">{result["answer"]}</div>',
                unsafe_allow_html=True,
            )
            if result["sources"]:
                render_sources(result["sources"])

    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
    )
    st.rerun()  # refresh so the suggestion buttons disappear once a conversation has started

with st.sidebar:
    st.header("About")
    st.write(
        "This assistant routes each question to melanoma literature (PubMed), drug data "
        "(PubChem/ChEMBL), or both, and remembers the current conversation so follow-up "
        "questions work naturally."
    )
    st.caption(f"Memory window: last {MAX_HISTORY_TURNS} exchanges")
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()
