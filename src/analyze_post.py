from webscrape import load_data, save_data

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

FREE_OPENROUTER_MODEL = "mistralai/devstral-small-2505:free" # Super lightweight model
MAX_CHARS = 10000 # approximately 2500 tokens

load_dotenv()

import os
from openai import OpenAI

def get_openrouter_client():
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    client = OpenAI(
        api_key=key,
        base_url="https://openrouter.ai/api/v1",
        timeout=90.0,
        max_retries=0,
    )

    return client

def prompt_ai(client, model=FREE_OPENROUTER_MODEL, messages=None, temperature=0, extra_headers={
        "HTTP-Referer": "https://github.com/mat-lee",
        "X-Title": "Advice Aggregator"
    }):
   """Returns a prompt given model and a message message"""
   return client.chat.completions.create(
      model=model,
      messages=messages,
      temperature=temperature,
      extra_headers=extra_headers,
  )

def gather_advice_and_group(client, doc: str):
  """Prompts a model to read text and identify pieces of advice."""

  # Setup prompt
  user = f"""
  Goal: From TEXT, extract each independent piece of advice verbatim and assign it to a theme.

  Return JSON only (no markdown/extras), exactly in this schema:
  {{
    "items": [
      {{"advice": string, "group": string}}
    ]
  }}

  Rules:
  - "advice": a verbatim substring of TEXT that is a self-contained actionable tip (~40–200 chars).
  - "group": a short label (≤3 words) describing the theme of that advice.
  - Use 2–6 distinct group labels overall.
  - If multiple advice items share the same theme, repeat the group name for each.
  - Do not include any extra text or keys outside the schema.

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

def get_alltext(df):
   # Combine title and body text from a dataframe
   # Returns a Series with combined text
   return "title: " + df['title'] + ". body: " + df['selftext'].fillna('')

def add_advice(client, raw_data):
    """Given a dataframe, returns a copy with every piece of advice and grouping"""
    data = raw_data.copy() # Create a copy

    # Clean and prepare the text
    data['alltext'] = get_alltext(data)

    data.drop(columns=["selftext", "title"], inplace=True)

    print(data['alltext'].head())

    def extract_items(client, text):
      text = text[:MAX_CHARS]
      res = gather_advice_and_group(client, text)        # call your LLM
      try:
          parsed = json.loads(res)             # parse string → dict
      except Exception as e:
          print("JSON parse error:", e, res)
          return []
      
      try:
         return parsed['items']
      except:
         return []

    # Add advice grouping dictionary pair to text
    data["advice_and_group"] = data["alltext"].map(lambda x: extract_items(client, x))
    
    # Get rid of alltext as its not necessary
    data.drop(columns="alltext", inplace=True)

    # Expand each advice/grouping pair per advice post
    data = data.explode("advice_and_group").reset_index(drop=True)

    # Drop na
    data.dropna(subset="advice_and_group", inplace=True)

    # Expand dictionary
    data["advice"] = data["advice_and_group"].map(lambda d: d.get("advice", "") if isinstance(d, dict) else "")
    data["group"] = data["advice_and_group"].map(lambda d: d.get("group", "") if isinstance(d, dict) else "")

    data.drop(columns="advice_and_group", inplace=True)

    # Return dataframe
    return data

def regroup_clusters(client, df):
  """Given a dataframe, modifies the data inplace and regroups the groupings"""
  # Get label counts to help the model choose canonical names
  counts = df['group'].value_counts().to_dict()

  # Build a small prompt
  user_prompt = f"""
  You are given a list of existing group labels with their frequencies.
  Return a JSON object that maps EACH old label to ONE consolidated canonical label.
  Keep labels short (1–3 words). Merge synonyms/redundant labels into a single canonical label.
  Output JSON only, no markdown or commentary.

  Example:
  Input counts: {{"Sleep": 42, "Sleep Schedule": 20, "Reading": 10}}
  Output: {{"Sleep":"Sleep", "Sleep Schedule":"Sleep", "Reading":"Reading"}}

  Input counts:
  {counts}
  """


  # Call the LLM
  completion = prompt_ai(
      client,
      messages=[{"role": "user", "content": user_prompt}],
  )

  raw = completion.choices[0].message.content.strip()

  if raw.startswith("```"):
      raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)   # remove opening ```json or ```
      raw = re.sub(r"```$", "", raw)               # remove closing ```
      raw = raw.strip()

  # Try parsing JSON safely
  try:
      mapping = json.loads(raw)
  except Exception as e:
      raise ValueError(f"Failed to parse JSON from model: {e}\nRaw output:\n{raw}")

  # Remap groupings
  df['group'] = df['group'].map(lambda g: mapping.get(g, g))

  return df

if __name__ == "__main__":
  raw_data = load_data("raw_data")
  raw_data = raw_data.iloc[0:10]

  client = get_openrouter_client()

  processed_data = add_advice(client, raw_data)

  regroup_clusters(client, processed_data)

  print(processed_data)

  # Save processed_data
  save_data(processed_data, "processed_data")