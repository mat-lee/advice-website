import re

import hdbscan
import numpy as np
import pandas as pd
import praw
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import umap

def save_data(df, name):
    df.to_csv(f"{name}.csv", index=False)

def load_data(name):
    df = pd.read_csv(f"{name}.csv")
    return df

def webscrape_reddit():
    data = {}

    def add_to_data(data, s):
        save_stats = ['id', 'selftext', 'subreddit_name_prefixed', 'title', 'ups', 'upvote_ratios']

        for stat in save_stats:
            if stat not in data:
                data[stat] = []
            data[stat].append(getattr(s, stat, None))
        
        return data

    # Initialize the Reddit client
    reddit = praw.Reddit(
        client_id="3mXT9JuzHHWNLYpFwHIkRw",
        client_secret="ezSxJjVDtXLob8UmKqj34l7uzVPNNQ",
        username="Subredditstick",
        password="97RedYoshis?",
        user_agent="my-scraper:v1.0 (by u/spoilsitck)"
    )

    for s in reddit.subreddit("getdisciplined").top(time_filter="all", limit=100):
        add_to_data(data, s)
    
    data = pd.DataFrame(data)

    return data

def split_sentences(text):
    # simple fallback; for better accuracy, use spaCy sentencizer
    if pd.isna(text):
        return []
    parts = re.split(r"(?<=[.!?])\s+\n?|\n{2,}", text)
    lines = [l.strip() for l in re.split(r"\n+", text)]
    return [x.strip() for x in parts + lines if x and len(x.strip()) > 5]

def is_line_short_or_tldr_or_link(line) -> bool:
    if pd.isna(line): 
        return False
    if len(line) < 10: return False
    if line.lower().startswith(("edit:", "source:", "tl;dr", "tldr")): return False
    if "http://" in line or "https://" in line: return False
    return True

def load_llm():
    return SentenceTransformer("all-MiniLM-L6-v2")

def label_cluster(texts):
    words = re.findall(r"[a-z][a-z]+", " ".join(texts).lower())
    words = [w for w in words if w not in ENGLISH_STOP_WORDS and len(w) > 3]
    # top 2 keywords
    from collections import Counter
    common = [w for w,_ in Counter(words).most_common(2)]
    return " / ".join(common) if common else "General"

def process_raw_data(raw_data):
    # This function is a placeholder for any additional processing needed on raw data
    data = raw_data.copy() # Create a copy

    # Clean and prepare the text
    data['alltext'] = data['title'] + ". " + data['selftext'].fillna('')

    data.drop(columns="selftext", inplace=True)

    print(data['alltext']).head()
    data["cleaned_text"] = raw_data["alltext"].apply(split_sentences)

    data.drop(columns="alltext", inplace=True)

    data = data.explode("cleaned_text").reset_index(drop=True)

    data = data[data["cleaned_text"].apply(is_line_short_or_tldr_or_link)]

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

if __name__ == "__main__":
    # raw_data = webscrape_reddit()
    # save_data(raw_data, "raw_data")

    raw_data = load_data("raw_data")
    data = process_raw_data(raw_data)
    # save_data(data, "processed_data")

    data = load_data("processed_data")

    sample = data.sample(n=100)
    sample.drop(columns=["id"], inplace=True)

    for index, row in sample.sort_values(by="cluster").iterrows():
        print(f"Cluster: {row['cluster_name']}\nText: {row['cleaned_text']}\n")