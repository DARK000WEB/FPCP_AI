# DARK@WEB FPCP – AI-Powered Scientific Article Generation Platform

<p align="center">
  <img src="https://img.shields.io/badge/Status-Production%20Ready-success?style=for-the-badge" alt="Status">
  <img src="https://img.shields.io/badge/Python-3.11%2B-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.115%2B-009688?style=for-the-badge&logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/Llama--3--70B--4bit-FF6F61?style=for-the-badge" alt="Llama-3-70B">
  <img src="https://img.shields.io/badge/FAISS%20%2B%20BM25-Hybrid%20Search-28a745?style=for-the-badge" alt="FAISS + BM25">
  <img src="https://img.shields.io/badge/Multi--Agent%20RAG-Advanced-00ffff?style=for-the-badge" alt="Multi-Agent">
</p>

<p align="center">
  <span style="font-size: 3.8em;">Brain</span><br>
  <strong style="font-size: 2.5em; background: linear-gradient(90deg, #00ffff, #0080ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
    DARK@WEB FPCP
  </strong><br>
  <i>Advanced Fully-Automatic Scientific Article Generation Platform Powered by AI</i>
</p>

## Project Overview

**DARK@WEB FPCP** is a state-of-the-art, fully automated platform that generates high-quality scientific papers, theses, journal articles, and academic content using next-generation artificial intelligence.

Powered by a Multi-Agent architecture, Retrieval-Augmented Generation (RAG), a 70B-parameter Llama-3 model fine-tuned with LoRA, and hybrid FAISS + BM25 retrieval, the system can produce a complete, original, citable, journal-ready scientific article in under 90 seconds.

Designed for PhD students, professors, researchers, and academic publishers.

## Key Features

| Feature                            | Description                                                                                   |
|------------------------------------|-----------------------------------------------------------------------------------------------|
| Multi-Agent Architecture           | Planner → Retriever → Generator → Critic → Post-Processor                                    |
| Hybrid Retrieval                   | Semantic search (FAISS L2) + Keyword search (BM25) + Real-time CrossRef API integration      |
| Llama-3-70B 4-bit + LoRA           | Ultra-high-quality text generation, fine-tuned on PubMed dataset                             |
| Adaptive Academic Level            | Automatically adjusts complexity (Bachelor → Master → PhD/Professor)                         |
| Built-in Quality Evaluation        | Critic Agent powered by DeBERTa + custom Coherence model                                      |
| Multi-Format Export                | PDF • Word • LaTeX • Markdown                                                               |
| Full Persian/English Translation   | Long-text chunked translation (no character limit)                                            |
| Automatic Figure & Chart Generation| Integrated Matplotlib-based visualization inserted into PDF/Word/LaTeX                      |
| Plagiarism-Controlled Output       | 100% original content with maximum source similarity ≤ 12%                                   |
| Internal + External Knowledge Base | Persistent article storage + live retrieval from CrossRef                                    |

## Technologies Stack

- **Backend**: FastAPI + SQLAlchemy 2.0 (Async)
- **Database**: PostgreSQL
- **Core Model**: Llama-3-70B-Instruct (4-bit quantized) + LoRA fine-tuning on PubMed
- **Embeddings**: `sentence-transformers/all-mpnet-base-v2`
- **Vector Search**: FAISS (L2 normalized) + BM25 fallback
- **External Knowledge**: CrossRef API
- **Critic Model**: `microsoft/deberta-v3-large` + Custom Coherence Head
- **Translation**: Google Translate API (chunked for long texts)
- **File Generation**: FPDF • python-docx • pylatexenc • Matplotlib
- **Authentication**: JWT + bcrypt

## Performance (Internal Benchmark – 100 Articles)

| Metric                        | Result                  |
|-------------------------------|-------------------------|
| Average ROUGE-L               | 0.72                    |
| Average Coherence Score       | 0.91                    |
| Max Similarity to Sources     | ≤ 12%                   |
| Generation Time (~4000 words) | 45–90 seconds           |

## Quick Start (Development)

```bash
git clone https://github.com/DARK000WEB/FPCP_AI.git
cd FPCP_AI

python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

pip install -r requirements.txt

cp .env.example .env
# Fill in DATABASE_URL, JWT_SECRET_KEY, etc.

alembic upgrade head

uvicorn main:app --reload --port=8000