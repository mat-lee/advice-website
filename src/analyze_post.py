from webscrape import load_data

import json
import re
import os

from dotenv import load_dotenv
import hdbscan
import numpy as np
from openai import OpenAI
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import umap

load_dotenv()  # reads .env in the working dir (won't override existing env by default)

def gather_advice(client, doc: str):
  user = f"""
  Task: If the text below contains multiple independent pieces of advice, (A) extract each advice item verbatim and (B) group related items by theme. If not a collection, return exactly {"{"}"type":"single"{"}"}.

  DEFINITIONS
  - "Advice item": a self-contained, actionable tip that stands on its own (often a bullet, numbered line, or imperative sentence).
  - "Collection": ≥3 distinct advice items or a clear list of tips.

  OUTPUT
  Return JSON exactly in this schema:
  {{
    "type": "collection",
    "items": [
      {{"text": string, "start": int, "end": int}}
    ],
    "groups": [
      {{"label": string, "item_indexes": [int, ...]}}
    ],
    "unassigned": [int, ...]
  }}

  ITEM RULES
  - Each item is a verbatim, contiguous substring of the original text (provide 0-based "start"/"end" offsets).
  - Prefer a single sentence that reads as one actionable tip.
  - Length target ≈ 40–200 chars. If the only suitable sentence is longer, include it in full (do not paraphrase).
  - If a bullet contains multiple distinct tips, extract multiple items (one per tip).
  - Exclude boilerplate (greetings, headers, credits) by not selecting them.
  - You may omit leading bullet/number markers and surrounding whitespace from the span; otherwise the item must match the original text exactly.

  GROUPING RULES
  - Create 2–6 coherent themes.
  - Use short labels (≤ 3 words), e.g., "Sleep hygiene", "Planning", "Burnout".
  - Use 0-based "item_indexes" that reference positions in "items".
  - Groups must be disjoint; put leftovers in "unassigned".
  - Prefer grouping by outcome/behavior (e.g., routines, environment, diet, mindset).

  TEXT
  ```{doc}```
  """


  completion = client.chat.completions.create(
    extra_body={},
    model="z-ai/glm-4.5-air:free",
    messages=[
      {
        "role": "user",
        "content": user
      }
    ]
  )

  return completion.choices[0].message.content

def load_client():
  api_key = os.getenv("OPENROUTER_API_KEY")

  client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
  )

  return client

def load_llm():
    return SentenceTransformer("all-MiniLM-L6-v2")

def get_alltext(df):
   # Combine title and body text from a dataframe
   # Returns a Series with combined text
   return "title: " + df['title'] + ". body: " + df['selftext'].fillna('')

def process_raw_data(raw_data):
    # This function is a placeholder for any additional processing needed on raw data
    data = raw_data.copy() # Create a copy

    # Clean and prepare the text
    data['alltext'] = get_alltext(data)

    data.drop(columns=["selftext", "title"], inplace=True)

    print(data['alltext']).head()

    client = load_client()

    def extract_items(text):
      res = gather_advice(client, text)        # call your LLM
      try:
          parsed = json.loads(res)             # parse string → dict
      except Exception as e:
          print("JSON parse error:", e, res)
          return []

      if parsed.get("type") == "collection":
          return parsed.get("items", [])
      else:
          return []  # single-theme posts: no items

    data["advice_agg"] = data["alltext"].map(lambda x: extract_items(client, x))
    data


    # Embeddings
    llm = load_llm()
    data["embeddings"] = llm.encode(data["cleaned_text"].tolist(), show_progress_bar=True, convert_to_numpy=True).tolist()

    # Clustering
    X = np.vstack(data["embeddings"].values)

    # UMAP dimensionality reduction
    reducer = umap.UMAP(n_neighbors=25, n_components=100, metric="cosine", random_state=42)
    X = reducer.fit_transform(X)

    # HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, metric="euclidean", cluster_selection_method="eom")
    labels = clusterer.fit_predict(X)

    data["cluster"] = labels

    # Cluster names
    clusters = data["cluster"].unique()
    cluster_labels = {
        cluster: label_cluster(data[data['cluster'] == cluster]['cleaned_text'].values) for cluster in clusters
    }

    data['cluster_name'] = data['cluster'].map(cluster_labels)

    data.drop(columns=["embeddings", "subreddit_name_prefixed", 'title', 'ups', 'upvote_ratios'], inplace=True)

    return data

def label_cluster(texts):
    words = re.findall(r"[a-z][a-z]+", " ".join(texts).lower())
    words = [w for w in words if w not in ENGLISH_STOP_WORDS and len(w) > 3]
    # top 2 keywords
    from collections import Counter
    common = [w for w,_ in Counter(words).most_common(2)]
    return " / ".join(common) if common else "General"

if __name__ == "__main__":
  data = load_data("raw_data")
  print(data.head())
  example_post = data.iloc[0]['selftext']

  client = load_client()
  
  res = gather_advice(client, example_post)

  print(res)