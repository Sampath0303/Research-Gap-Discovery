# Research Gap Discovery Engine

An AI-powered platform for analyzing research papers, extracting limitations, clustering related findings, and generating structured research gaps with actionable ideas.

## Problem Statement

Researchers spend significant time manually reviewing papers and identifying research gaps.

## Solution

Research Gap Discovery Engine automates large parts of that workflow. It:

- Loads research papers from PDF files
- Extracts key information from each paper
- Identifies research limitations
- Clusters related limitations across papers
- Discovers research gaps
- Generates potential research ideas
- Provides semantic search using RAG

## Architecture

### RAG Pipeline

`PDFs -> Chunking -> Embeddings -> FAISS -> Retrieval -> Gemini RAG`

### Research Gap Pipeline

`PDFs -> Information Extraction -> Limitation Mining -> Clustering -> Research Gap Generation`

## Features

- Multi-PDF Analysis
- Semantic Search
- Research Gap Discovery
- Analytics Dashboard
- Export Functionality

## Tech Stack

- Python
- Streamlit
- LangChain
- FAISS
- Sentence Transformers
- Gemini API
- Scikit-Learn

## Project Structure

```text
ResearchGapDiscoveryEngine/
|-- data/
|   `-- papers/
|       |-- ALBERT.pdf
|       |-- Attention is All you need.pdf
|       |-- BERT.pdf
|       |-- DEBERTA.pdf
|       |-- DistilBERT.pdf
|       |-- ELECTRA.pdf
|       |-- GPT-2.pdf
|       |-- GPT-3.pdf
|       |-- RoBERTa.pdf
|       `-- T5_Transforme.pdf
|-- outputs/
|   |-- ALBERT.json
|   |-- Attention is All you need.json
|   |-- BERT.json
|   |-- DEBERTA.json
|   |-- DistilBERT.json
|   |-- ELECTRA.json
|   |-- GPT-2.json
|   |-- GPT-3.json
|   |-- RoBERTa.json
|   `-- research_gaps.json
|-- src/
|   |-- batch_extractor.py
|   |-- chunker.py
|   |-- cluster_limitations.py
|   |-- cluster_summarize.py
|   |-- config.py
|   |-- dashboard_utils.py
|   |-- extractor.py
|   |-- gap_summarize.py
|   |-- limitation_minor.py
|   |-- models.py
|   |-- pdf_loader.py
|   |-- rag.py
|   |-- research_gap_generator.py
|   |-- retriever.py
|   `-- vector_store.py
|-- streamlit_app.py
|-- requirements.txt
`-- README.md
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
```

### 3. Run the Streamlit app

```bash
streamlit run streamlit_app.py
```

## Outputs

### Extracted Paper JSON

Each paper output contains extracted sections such as methods, datasets, limitations, and future work.

### Structured Research Gaps JSON

`outputs/research_gaps.json` now uses this format:

```json
{
  "cluster": 0,
  "theme": "Scalable and Coherent Language Models",
  "limitations": ["..."],
  "research_gap": "Structured gap summary",
  "potential_idea": "Structured research idea"
}
```

## Dashboard Pages

- `Home`: high-level overview and key metrics
- `Paper Search`: semantic search with Gemini-powered answers
- `Limitations`: extracted limitation explorer
- `Clusters`: cluster-level theme and idea summaries
- `Research Gaps`: expandable cluster cards with JSON and CSV export
- `Analytics`: metrics and charts for papers, limitations, clusters, and gaps

## Future Scope

- 500+ paper processing
- Vector databases
- Automatic literature review generation
- Citation graph analysis

## Screenshots

Add project screenshots here after running the app:

- `docs/screenshots/home.png`
- `docs/screenshots/research-gaps.png`
- `docs/screenshots/analytics.png`

## Notes

- Keep your paper PDFs in `data/papers/`
- Structured outputs are written to `outputs/`
- The Research Gaps page reads directly from `outputs/research_gaps.json`
- JSON and CSV exports are available from the dashboard


Research Gap Discovery Engine is an AI-powered research assistant that helps researchers analyze multiple research papers, identify recurring limitations, discover unexplored research opportunities, and generate potential research ideas automatically.

Instead of manually reading dozens of papers to identify common challenges and future directions, the system combines Retrieval-Augmented Generation (RAG), semantic search, clustering, and large language models to accelerate the literature review process.
