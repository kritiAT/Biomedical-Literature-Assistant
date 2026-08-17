# 🧬 Biomedical Research Copilot

An AI-powered research assistant built with **RAG, LLMs, and multimodal document retrieval**, that helps researchers, bioinformaticians and healthcare professionals explore biomedical literature and drug information through natural language questions. The application provides two specialized assistants for exploring melanoma literature and analyzing user-provided research articles.

### Overview

Biomedical researchers often need to search through large volumes of scientific literature to find relevant information, compare findings, and extract evidence from research papers. **Biomedical Research Copilot** aims to simplify this process through retrieval-augmented generation (RAG).

The application provides two research assistants:

* **MEL-AI** — a melanoma-focused research assistant using a curated biomedical knowledge base.
* **BioLit-AI** — a custom literature assistant that allows users to upload PDFs or provide PMCIDs and ask questions about the selected articles.

The system retrieves relevant scientific literature and drug information, then generates evidence-backed answers with citations.



## 🚀 Live Demo

Try the deployed: **[Biomedical Research Copilot](https://biomedical-literature-assistant.streamlit.app/)**

> The app is deployed on Streamlit Cloud. Processing time may vary depending on the size and complexity of uploaded research articles.

---

## 🔍 Mel-AI — Melanoma Research Assistant

Mel-AI answers melanoma-specific research questions by retrieving from a pre-built knowledge
base of PubMed literature and drug data (ChEMBL + PubChem) and generating cited answers. 

* *Literature & drug search* with PMID, ChEMBL, and PubChem citations
* *Pinecone vector store* with separate literature and drug namespaces
* *Smart LLM routing* to query literature, drug data, or both
* *Conversational memory* for contextual follow-up questions
* *Scope-limited* to melanoma for focused, relevant responses

**Knowledge base:**

| Source | Downloaded/Extracted | After Cleaning | Chunks in Pinecone |
|---|---|---|---|
| Literature (PubMed) | 5,000 abstracts | 4,783 | 15,315 |
| Drugs (ChEMBL + PubChem) | 100 ChEMBL IDs | 57 | 113 |

**Pipeline:** NCBI/ChEMBL/PubChem ingestion → cleaning & chunking → OpenAI embeddings → Pinecone → LLM routing → grounded `gpt-4o-mini` responses with source citations.

> **Not medical advice:** For research and literature-review purposes only.



## 🤖 BioLit-AI — Biomedical Literature Assistant

A *multimodal RAG assistant* for querying and comparing up to 10 biomedical research papers from PDFs or PubMed Central (PMCID).

* *Document ingestion:* PDFs via Unstructured; PMC articles via NCBI E-utilities
* *Multimodal extraction:* text, tables, figures, and section metadata
* *Semantic retrieval:* summarized content embedded in ChromaDB
* *Source-grounded answers:* retrieves raw content for accurate citations

**Pipeline:** PDF/PMC ingestion → content extraction → chunking & summarization → Chroma indexing → MultiVectorRetriever → cited RAG responses.

**Key design:** summaries improve semantic retrieval, while original content is preserved for accurate citations.

> **Intended use:** MVP research-reading and literature-review aid; not a diagnostic or clinical decision-support tool.

--- 

## 🏗️ System Architecture

                         Streamlit UI
                              │
                 ┌────────────┴────────────┐
                 │                         │
              MEL-AI                   BioLit-AI
        Melanoma Research          Custom Literature
            Assistant                 Assistant
                 │                         │
                 ↓                         ↓
        User Research Query        PDF Upload / PMCID
                 │                         │
                 ↓                         ↓
        Melanoma Knowledge       Document Processing
             Base                          |
                 │                         │
                 ↓                         ↓
          Pinecone Vector DB      Cleaning & Chunking
                 │                         │
                 │                         ↓
                 │                  Summarization
                 │                         │
                 │                         ↓
                 │                  Chroma Vector DB
                 │                         │
                 └────────────┐    ┌───────┘
                              ↓    ↓
                         RAG Retrieval
                              │
                              ↓
                          OpenAI LLM
                              │
                              ↓
                    Grounded Answer + Sources
                              │
                              ↓
                         Streamlit UI


----

## Project Structure

```text
Biomedical-Literature-Assistant/
│
├── Home.py                     # Main Streamlit entry point
├── rag_backend.py              # Core RAG pipeline and LLM logic
├── requirements.txt            # Python dependencies
├── packages.txt                # System-level dependencies for Streamlit Cloud
├── README.md                   # Project documentation
│
├── pages/                      # Streamlit multi-page application
│   ├── 01_Mel-AI.py            # Melanoma research assistant
│   └── 02_BioLit-AI.py         # Biomedical literature assistant
│
├── notebooks/                  # Knowledge base preparation
│   ├── 01_literature_embeddings.ipynb
│   └── 02_drug_embeddings.ipynb
│
├── files/                      # Presentation and project materials
│
└── data/                       # Datasets and processed knowledge-base files
```

## Setup

**Requirements**

* Python **3.12**


**Dependencies**

All required Python packages and versions are listed in **`requirements.txt`**.

```
pip install -r requirements.txt
```

**Environment variables** (`.env` file or host secrets):
```
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key        # Mel-AI only
NCBI_EMAIL=you@example.com                    # for PubMed/PMC fetches
NCBI_API_KEY=                                 # optional, raises Entrez rate limit
```

**Run Locally**

```bash
streamlit run Home.py
```

---

## Current Limitations

* **Limited knowledge base:** MEL-AI currently focuses primarily on melanoma and a selected set of drugs.
* **PDF/PMC processing time:** BioLit-AI can take significant time to parse and index large or complex research articles.
* **Resource constraints:** The application is designed as an MVP and uses free/limited cloud resources.
* **Knowledge coverage:** The assistant should not be considered a comprehensive biomedical database.

## Future Improvements

* Expand the biomedical knowledge base beyond melanoma.
* Improve PDF ingestion speed through asynchronous/background processing.
* Add persistent vector storage for uploaded documents.
* Improve table and figure understanding.
* Add PubMed search directly into the application.
* Introduce evaluation datasets for measuring RAG retrieval and answer quality.

## Disclaimer

This project is intended as a **research and educational tool**. Generated responses may contain errors and should be verified against the original scientific literature. It is not intended to provide medical diagnosis or treatment recommendations.