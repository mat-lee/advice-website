<div align="center">
  
# 🧭 Advice Aggregator (NLP)
**From long posts ➜ clean, clustered, browsable tips.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](#)
[![Sentence-Transformers](https://img.shields.io/badge/Embeddings-SentenceTransformers-4E9A06)](#)
[![Clustering](https://img.shields.io/badge/Clustering-HDBSCAN-7B1FA2)](#)
[![UMAP](https://img.shields.io/badge/Viz-UMAP--learn-0E7C86)](#)
[![License](https://img.shields.io/badge/License-MIT-black)](#license)

</div>

---

## ✨ Highlights
- Scrape advice-heavy posts (e.g., Reddit) → **CSV**
- LLM-assisted **verbatim** tip extraction
- Sentence embeddings → **HDBSCAN** clustering (optional **UMAP** for viz)
- Exports **JSON/CSV** ready for a React/Streamlit UI

---

## 🚀 Quickstart

```bash
# 1) Setup
python -m venv .venv && source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a `.env` (never commit secrets):
```dotenv
OPENAI_API_KEY=sk-...
REDDIT_CLIENT_ID=...
REDDIT_CLIENT_SECRET=...
REDDIT_USERNAME=...
REDDIT_PASSWORD=...
REDDIT_USER_AGENT=advice-aggregator:v1.0 (by u/yourname)

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
OPENAI_MODEL=gpt-4o-mini
```

```bash
# 2) Scrape → raw_data.csv
python webscrape.py

# 3) Analyze & cluster
python analyze_post.py
```

---

## 🧩 Stack
- `praw`, `python-dotenv`, `pandas`, `numpy`
- `openai` (>= 1.0), `sentence-transformers`
- `hdbscan`, `umap-learn`, `scikit-learn`

> Tip: If you see `module 'umap' has no attribute 'UMAP'`, install **`umap-learn`** (not `umap`).

---

## 📁 Project Structure
```
.
├── analyze_post.py
├── webscrape.py
├── requirements.txt
├── .env.example
└── artifacts/           # generated
```

---

## ✅ Notes
- Respect site ToS and rate limits; cache locally.

---

## 📜 License
MIT
