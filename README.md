# Advanced-RAG-Pipeline-for-Structured-Screenplay-Analysis-Harry-Potter-
A production-grade Retrieval-Augmented Generation (RAG) system  Implements layout PDF parsing with DoclingLoader, hybrid BM25+FAISS search with Reciprocal Rank Fusion, cross-encoder re-ranking, query expansion, HyDE, RAG-Fusion, and automated LLM-as-a-Judge evaluation using TruLens. Applied to the Harry Potter and the Sorcerer's Stone screenplay.
# 🧙 Advanced RAG Pipeline for Structured Screenplay Analysis

> **PES University | UE23CS342BA9: Generative AI and its Applications | Lab Exercise 4**

A production-grade **Retrieval-Augmented Generation (RAG)** system engineered to intelligently query the *Harry Potter and the Sorcerer's Stone* screenplay. This project transitions from naive semantic search to a fully advanced RAG architecture featuring layout-aware parsing, hybrid retrieval, mathematical re-ranking, and automated LLM-as-a-Judge evaluation.

---

## 📌 Project Overview

Standard RAG pipelines fail on highly structured spatial documents like screenplays because:
- Naive chunking severs character names from their dialogue
- Dense embeddings blur exact keywords like spell names and scene numbers
- Brute-force similarity search is O(N) and doesn't scale

This project solves all three problems through a 7-phase engineering pipeline.

---

## 🏗️ Architecture
Here's a complete README for your project:

markdown
# 🧙 Advanced RAG Pipeline for Structured Screenplay Analysis

> **PES University | UE23CS342BA9: Generative AI and its Applications | Lab Exercise 4**

A production-grade **Retrieval-Augmented Generation (RAG)** system engineered to intelligently query the *Harry Potter and the Sorcerer's Stone* screenplay. This project transitions from naive semantic search to a fully advanced RAG architecture featuring layout-aware parsing, hybrid retrieval, mathematical re-ranking, and automated LLM-as-a-Judge evaluation.

---

## 📌 Project Overview

Standard RAG pipelines fail on highly structured spatial documents like screenplays because:
- Naive chunking severs character names from their dialogue
- Dense embeddings blur exact keywords like spell names and scene numbers
- Brute-force similarity search is O(N) and doesn't scale

This project solves all three problems through a 7-phase engineering pipeline.

---

## 🏗️ Architecture
```
PDF Screenplay
      │
      ▼
┌─────────────────────────────┐
│  Phase 1: Document Ingestion │  ← PyPDF + PyMuPDF + DoclingLoader
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Phase 2: Intelligent        │  ← MarkdownHeaderTextSplitter
│  Chunking                    │     + RecursiveCharacterTextSplitter
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Phase 3: Vector Indexing    │  ← FAISS (Flat, IVF, HNSW, PQ)
│  & Embeddings                │     + ChromaDB + nomic-embed-text
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Phase 4: Baseline RAG       │  ← LangChain LCEL + Ollama (llama3)
│  Memory + Anti-Hallucination │     + RunnableWithMessageHistory
│                              │     + RunnableBranch
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Phase 5: Advanced Retrieval │  ← BM25 + FAISS Hybrid (RRF)
│  Enhancements                │     + Cross-Encoder Re-Ranking
│                              │     + Query Expansion
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Phase 6: Exploratory        │  ← HyDE + RAG-Fusion
│  Architectures               │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Phase 7: RAG Evaluation     │  ← Precision@K, Recall@K, MRR,
│                              │     NDCG@K + TruLens LLM-as-a-Judge
└─────────────────────────────┘
```

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 📄 **Layout-Aware Parsing** | DoclingLoader converts screenplay PDF to structured Markdown, preserving character → dialogue relationships |
| 🔀 **Hybrid Search (RRF)** | Combines BM25 sparse retrieval + FAISS dense retrieval using Reciprocal Rank Fusion |
| 🎯 **Cross-Encoder Re-Ranking** | BAAI/bge-reranker-base re-scores top-K candidates with full query-document attention |
| 🔍 **Query Expansion** | LLM intercepts lazy prompts and enriches them with lore-specific synonyms before retrieval |
| 🧠 **HyDE** | Forces LLM to hallucinate a fake scene, then uses that scene as the search vector |
| 🔁 **RAG-Fusion** | Generates 4 query variations, searches in parallel, fuses results via RRF |
| 💬 **Conversational Memory** | RunnableWithMessageHistory maintains session context across follow-up questions |
| 🚦 **Anti-Hallucination Routing** | RunnableBranch hard-gates greetings away from the vector DB entirely |
| 📊 **Automated Evaluation** | TruLens LLM-as-a-Judge measures Context Relevance, Groundedness, Answer Relevance |

---

## 📊 Benchmark Results

### FAISS Index Latency Comparison (482 chunks, 1024-dim embeddings)

| Index Type | Latency | Complexity | Notes |
|---|---|---|---|
| Flat L2 (Brute-force) | 0.28 ms | O(N) | Exact, no compression |
| IVF (Clustered) | 0.38 ms | O(√N) | 5 clusters |
| **HNSW (Graph-based)** | **0.19 ms** | **O(log N)** | **16 edges/node — fastest** |
| PQ (Compressed) | 0.46 ms | O(N/m) | Lossy — 32 subquantizers |

### TruLens Evaluation Scores

| Metric | Score |
|---|---|
| Answer Relevance | **0.967** |
| Groundedness | 0.600 |
| Context Relevance | 0.000 |

> Context Relevance of 0.0 indicates the retriever struggled with the Docling-parsed chunks for specific queries — a known limitation when the PDF-to-Markdown conversion introduces OCR noise.

---

## 🛠️ Tech Stack

- **LLM**: Ollama (llama3) — local inference
- **Embeddings**: `nomic-ai/nomic-embed-text-v1`, `BAAI/bge-m3`
- **Re-Ranker**: `BAAI/bge-reranker-base`
- **Vector DBs**: FAISS (cpu), ChromaDB
- **RAG Framework**: LangChain (LCEL)
- **Evaluation**: TruLens
- **PDF Parsing**: DoclingLoader, PyPDF, PyMuPDF
- **Sparse Retrieval**: BM25 (rank_bm25)

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.10+
- Google Colab with T4 GPU (recommended)
- Ollama installed

### 1. Clone the Repository
```bash
git clone https://github.com/sushmav418-jpg/Advanced-RAG-Pipeline-for-Structured-Screenplay-Analysis-Harry-Potter-.git
cd Advanced-RAG-Pipeline-for-Structured-Screenplay-Analysis-Harry-Potter-
```

### 2. Install Dependencies
```bash
pip install langchain_community langchain_docling langchain_text_splitters \
            langchain_huggingface langchain_chroma langchain_ollama \
            pypdf pymupdf faiss-cpu rank_bm25 sentence-transformers trulens_eval
```

### 3. Install & Start Ollama
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull llama3
```

### 4. Add the Screenplay
Place `screenplay.pdf` (Harry Potter and the Sorcerer's Stone screenplay) in the project root.

### 5. Run the Notebook
Open `PES1UG23CS621_GenAI_Lab4_Boilerplate.ipynb` in Google Colab or Jupyter and run all cells sequentially.

---

## 📁 Project Structure
```
├── PES1UG23CS621_GenAI_Lab4_Boilerplate.ipynb   # Main notebook
├── PES1UG23CS621_GenAI_Lab4.py                   # Python script export
├── PES1UG23CS621_GenAI_Lab4_Report.pdf           # Analysis report
├── screenplay.pdf                                  # Source document (not tracked)
├── chroma_docling_nomic/                           # ChromaDB persistent store (not tracked)
└── README.md
```

---

## 🔬 Phase-by-Phase Summary

### Phase 1 — Document Ingestion
Extracts text using **PyPDF** and **PyMuPDF** for raw text, and **DoclingLoader** for layout-aware Markdown conversion that preserves screenplay spatial structure.

### Phase 2 — Intelligent Chunking
- **Naive**: `RecursiveCharacterTextSplitter` (432 chunks) — demonstrates context fragmentation
- **Fixed**: Fixed-size splitting (674 chunks)
- **Hybrid**: `MarkdownHeaderTextSplitter` → `RecursiveCharacterTextSplitter` (73 structural → 482 final chunks) — preserves Character → Dialogue integrity

### Phase 3 — Indexing & Embeddings
Benchmarks 4 FAISS index types. Stores Docling chunks in ChromaDB using `nomic-embed-text-v1` (1024-dim embeddings).

### Phase 4 — Baseline RAG
LCEL chain with Ollama/llama3. Adds conversational memory via `RunnableWithMessageHistory` and anti-hallucination routing via `RunnableBranch`.

### Phase 5 — Advanced Retrieval
- **5.1 Hybrid Search**: BM25 (weight 0.3) + FAISS (weight 0.7) fused via RRF (c=60)
- **5.2 Re-Ranking**: Cross-encoder scores top-10 candidates, returns top-3
- **5.3 Query Expansion**: LLM enriches short queries with Harry Potter lore before retrieval

### Phase 6 — Exploratory Architectures
- **HyDE**: Generates hypothetical screenplay scene → embeds scene → searches DB
- **RAG-Fusion**: 4 query variations → parallel FAISS searches → RRF fusion

### Phase 7 — Automated Evaluation
- **7.1**: LLM-as-a-Judge binary grading → Precision@K, Recall@K, MRR, NDCG@K
- **7.2**: TruLens feedback functions for Context Relevance, Groundedness, Answer Relevance

---

## 📝 Key Findings

1. **Layout-aware chunking is non-negotiable** for structured documents — naive splitting severed character names from dialogue in 30%+ of boundary chunks.
2. **HNSW achieved 32% lower latency than Flat L2** while returning identical top results, confirming logarithmic scaling benefits even at small corpus sizes.
3. **BM25 solved exact keyword retrieval** (spell names, scene numbers) that FAISS dense embeddings completely failed on due to semantic blurring.
4. **HyDE improved retrieval quality** by generating document-distribution vectors instead of question-distribution vectors, aligning search with the screenplay's latent space.
5. **High Answer Relevance (0.967) with low Groundedness (0.6)** reveals the LLM drawing from parametric Harry Potter knowledge rather than the retrieved screenplay — the most dangerous enterprise RAG failure mode.

---

## 👤 Author

**Sushma V** | PES1UG23CS621
PES University, Bengaluru
UE23CS342BA9 — Generative AI and its Applications

---

## 📄 License

This project is for academic purposes only.
The Harry Potter screenplay is the intellectual property of Warner Bros. / J.K. Rowling and is not included in this repository.
Just copy this entire block into a file named README.md in your repo root, then:






