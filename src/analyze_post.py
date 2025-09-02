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

def prompt_ai(client, model="z-ai/glm-4.5-air:free", messages=None, temperature=0):
   return client.chat.completions.create(
      model="z-ai/glm-4.5-air:free",
      messages=messages,
      temperature=0
  )

def gather_advice(client, doc: str):
  user = f"""
  Task: If the text below contains multiple independent pieces of advice, (A) extract each advice item verbatim and (B) group related items by theme. If not a collection, return exactly {{"type":"single"}}.

  DEFINITIONS
  - "Advice item": a self-contained, actionable tip that stands on its own (often a bullet, numbered line, or imperative sentence).
  - "Collection": ≥3 distinct advice items or a clear list of tips.

  OUTPUT
  Return JSON exactly in this schema, with no extra text, no markdown formatting, and no explanations.
  Schema (keys must appear exactly as shown):
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
  - Each item must be a verbatim, contiguous substring of the original text (0-based "start"/"end" offsets; "end" is exclusive).
  - Prefer a single sentence that reads as one actionable tip.
  - Target length ≈ 40–200 chars; if the only suitable sentence is longer, include it whole (no paraphrasing).
  - If a bullet contains multiple distinct tips, extract multiple items (one per tip).
  - Exclude boilerplate (greetings, headers, credits) by not selecting them.
  - You may omit leading bullet/number markers and surrounding whitespace; otherwise the item must match the original exactly.

  GROUPING RULES
  - Create 2–6 coherent, disjoint themes.
  - Labels ≤ 3 words (e.g., "Sleep hygiene", "Planning", "Burnout").
  - Use 0-based "item_indexes" that reference positions in "items".
  - Any item not placed in a group must be referenced by index in "unassigned".
  - Prefer grouping by outcome/behavior (routines, environment, diet, mindset).

  VALIDATION
  - If the text is not a collection, output exactly: {{"type":"single"}} (no other keys).
  - If it is a collection, output exactly one JSON object with the schema above.
  - Never output markdown fences or explanatory text.

  TEXT
  {doc}
  """

  completion = prompt_ai(
     client, 
     messages=[
        {"role": "system", "content": "You are a careful extraction engine. Output must be strictly valid JSON. Do not include markdown fences, code blocks, or commentary. Start your response with \"{\" and end it with \"}\"."},
        {"role": "user", "content": user}
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

def add_advice(raw_data):
    # This function is a placeholder for any additional processing needed on raw data
    data = raw_data.copy() # Create a copy

    # Clean and prepare the text
    data['alltext'] = get_alltext(data)

    data.drop(columns=["selftext", "title"], inplace=True)

    print(data['alltext'].head())

    client = load_client()

    def extract_items(client, text):
      res = gather_advice(client, text)        # call your LLM
      try:
          parsed = json.loads(res)             # parse string → dict
      except Exception as e:
          print("JSON parse error:", e, res)
          return []

      if parsed.get("type") != "collection":
          return []

      items = parsed.get("items", [])
      groups = parsed.get("groups", [])

      # Map item indexes to group labels
      group_map = {}
      for g in groups:
          for i in g.get("item_indexes", []):
              group_map.setdefault(i, []).append(g["label"])

      # Merge advice items with their groups
      merged = []
      for idx, item in enumerate(items):
          merged.append({
              "advice": item["text"],
              "grouping": group_map.get(idx, [])
          })
        
      return merged

    # Add advice grouping dictionary pair to text
    data["advice_agg"] = data["alltext"].map(lambda x: extract_items(client, x))
    data.drop(columns="alltext", inplace=True)
    data = data.explode("advice_agg").reset_index(drop=True)

    # Drop na
    data.dropna(subset="advice_agg", inplace=True)

    # Expand dictionary
    data["advice"] = data["advice_agg"].map(lambda d: d.get("advice", "") if isinstance(d, dict) else "")
    data["grouping_list"] = data["advice_agg"].map(lambda d: d.get("grouping", []) if isinstance(d, dict) else [])

    # Return dataframe
    return data

def regroup_clusters(client, df):
  """Given a dataframe, regroups the groupings and modifies in place."""
  advice = df['advice']
  groupings = df['grouping']

  pairings = list(zip(advice, groupings))

  # Build the message to send to the LLM
  user_prompt = f"""
  You are given a list of (advice_text, grouping_label) pairs.

  Your job: review the labels and return a JSON object mapping each old grouping label
  to a new, consolidated label. 

  Guidelines:
  - Groupings should be short (1–3 words).
  - Merge synonymous or redundant labels into one.
  - Make labels clear and intuitive.
  - Only output the JSON mapping, no explanation, no markdown.

  Example:
  Input: [("Go to bed early","Sleep"), ("Set alarm","Sleep Schedule"), ("Read books","Reading")]
  Output: {{"Sleep Hygiene":"Sleep","Sleep Schedule":"Sleep"}}

  Now process this data:
  {pairings}
  """

  # Call the LLM
  completion = prompt_ai(
      client,
      messages=[{"role": "user", "content": user_prompt}],
  )

  raw = completion.choices[0].message.content.strip()

  # Try parsing JSON safely
  try:
      mapping = json.loads(raw)
  except Exception as e:
      raise ValueError(f"Failed to parse JSON from model: {e}\nRaw output:\n{raw}")

  # Remap groupings
  df['grouping'] = df['grouping'].map(lambda g: mapping.get(g, g))

  return df

if __name__ == "__main__":
  raw_data = load_data("raw_data")
  raw_data = raw_data.iloc[0:1]
  processed_data = add_advice(raw_data)
  print(processed_data)

  # Save processed_data
  processed_data.to_csv("p2.csv", index=False)