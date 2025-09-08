from webscrape import load_data, save_data

import json
import re
import os
import time

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

# Planned features: make a new file that shows whether or not a specific post id has been processed to find advice or not.
# Allow the option for it a be a continuous process; it automatically adds to process_data so if there's an interruption it won't lose progress.

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
        max_retries=2,  # Increased retries
    )

    return client

def sanitize_json_string(raw: str) -> str:
    """More robust JSON cleaning"""
    if not raw or not isinstance(raw, str):
        return "{}"
    
    # Strip markdown code fences like ```json ... ```
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"```$", "", raw)
    
    # Find the first { and last } to extract just the JSON part
    start = raw.find('{')
    end = raw.rfind('}')
    if start != -1 and end != -1 and end > start:
        raw = raw[start:end+1]
    
    return raw.strip()

def prompt_ai(client, model=FREE_OPENROUTER_MODEL, messages=None, temperature=0, extra_headers={
        "HTTP-Referer": "https://github.com/mat-lee",
        "X-Title": "Advice Aggregator"
    }):
    """Returns a prompt given model and a message with better error handling"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            extra_headers=extra_headers,
        )
        return response
    except Exception as e:
        print(f"API Error: {e}")
        return None

def gather_advice_and_group(client, doc: str):
    """Prompts a model to read text and identify pieces of advice with better error handling."""
    
    # Clean the document text to avoid JSON issues
    doc_clean = doc.replace('"', "'").replace('\n', ' ').replace('\r', ' ')
    doc_clean = ' '.join(doc_clean.split())  # Normalize whitespace

    # Setup prompt with more explicit instructions
    user = f"""
Goal: From TEXT, extract each independent piece of advice verbatim and assign it to a theme.

Return JSON only (no markdown/extras), exactly in this schema:
{{
    "items": [
        {{"advice": "example advice here", "group": "Theme Name"}}
    ]
}}

Rules:
- "advice": a verbatim substring of TEXT that is a self-contained actionable tip (40-200 chars).
- Replace all double quotes in advice text with single quotes
- "group": a short label (≤3 words) describing the theme of that advice.
- Every "advice" must make complete sense on its own
- Use 2-6 distinct group labels overall.
- If no clear advice found, return {{"items": []}}
- Remove any formatting symbols like **bold**, *italics*, [links](url)
- Do not include any extra text outside the schema.

TEXT: {doc_clean[:MAX_CHARS]}
"""

    completion = prompt_ai(
        client, 
        messages=[
            {"role": "system", "content": "You are a careful extraction engine. Output must be strictly valid JSON. Do not include markdown fences, code blocks, or commentary. Start your response with '{' and end it with '}'."},
            {"role": "user", "content": user}
        ]
    )

    if not completion or not completion.choices:
        print("No completion received from API")
        return {"items": []}

    raw = completion.choices[0].message.content
    if not raw:
        print("Empty response from API")
        return {"items": []}
        
    raw = raw.strip()
    raw = sanitize_json_string(raw)

    try:
        parsed = json.loads(raw)
        # Validate the structure
        if not isinstance(parsed, dict) or "items" not in parsed:
            print(f"Invalid JSON structure: {parsed}")
            return {"items": []}
        return parsed
    except Exception as e:
        print(f"Failed to parse JSON in gather_advice_and_group: {e}")
        print(f"Raw output:\n{raw[:500]}...")  # Show first 500 chars
        return {"items": []}

def get_alltext(df):
    """Combine title and body text from a dataframe"""
    # Handle missing values better
    title_text = df['title'].fillna('').astype(str)
    body_text = df['selftext'].fillna('').astype(str)
    
    # Only include title prefix if title exists
    combined = []
    for title, body in zip(title_text, body_text):
        parts = []
        if title.strip():
            parts.append(f"title: {title}")
        if body.strip():
            parts.append(f"body: {body}")
        combined.append(". ".join(parts))
    
    return pd.Series(combined, index=df.index)

def add_advice(client, raw_data):
    """Given a dataframe, returns a copy with every piece of advice and grouping"""
    data = raw_data.copy()
    print(f"Starting with {len(data)} rows")

    # Clean and prepare the text
    data['alltext'] = get_alltext(data)
    
    # Filter out rows with very little text (less than 50 characters)
    data = data[data['alltext'].str.len() >= 50].copy()
    print(f"After filtering short texts: {len(data)} rows")

    # Keep original columns but work with alltext
    # Don't drop selftext/title yet in case we need them for debugging

    def extract_items(client, text, row_index):
        """Extract advice items with better error handling and logging"""
        if not text or len(text.strip()) < 50:
            return []
            
        text = text[:MAX_CHARS]
        try:
            parsed = gather_advice_and_group(client, text)
            items = parsed.get("items", [])
            print(f"Row {row_index}: Found {len(items)} advice items")
            return items
        except Exception as e:
            print(f"Error processing row {row_index}: {e}")
            return []

    # Add advice grouping with row index for debugging
    print("Extracting advice from posts...")
    data["advice_and_group"] = [
        extract_items(client, text, idx) 
        for idx, text in enumerate(data["alltext"])
    ]
    
    # Count how many rows have advice
    has_advice = data["advice_and_group"].apply(lambda x: len(x) > 0 if x else False)
    print(f"Rows with advice found: {has_advice.sum()}/{len(data)}")

    # Drop unnecessary columns
    data.drop(columns=["alltext", "selftext", "title"], inplace=True)

    # Expand each advice/grouping pair per advice post
    data = data.explode("advice_and_group").reset_index(drop=True)
    print(f"After exploding advice: {len(data)} rows")

    # Drop rows where advice_and_group is empty list or None
    data = data[data["advice_and_group"].apply(lambda x: x is not None and x != [] and isinstance(x, dict))].copy()
    print(f"After filtering valid advice: {len(data)} rows")

    if len(data) == 0:
        print("WARNING: No valid advice found in any posts!")
        return data

    # Expand dictionary safely
    data["advice"] = data["advice_and_group"].apply(lambda d: d.get("advice", "") if isinstance(d, dict) else "")
    data["group"] = data["advice_and_group"].apply(lambda d: d.get("group", "") if isinstance(d, dict) else "")

    # Filter out empty advice
    data = data[data["advice"].str.len() > 0].copy()
    print(f"Final advice count: {len(data)} pieces of advice")

    data.drop(columns=["advice_and_group"], inplace=True)

    return data

def regroup_clusters(client, df):
    """Given a dataframe, modifies the data inplace and regroups the groupings"""
    if len(df) == 0:
        print("No data to regroup")
        return df
        
    # Get label counts to help the model choose canonical names
    counts = df['group'].value_counts().to_dict()
    print(f"Original groups: {counts}")

    # Build a small prompt
    user_prompt = f"""
You are given a list of existing group labels with their frequencies.
Return a JSON object that maps EACH old label to ONE consolidated canonical label.
Keep labels short (1-3 words). Merge synonyms/redundant labels into a single canonical label.
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

    if not completion or not completion.choices:
        print("Failed to get regrouping response, keeping original groups")
        return df

    raw = completion.choices[0].message.content
    if not raw:
        print("Empty regrouping response, keeping original groups")  
        return df
        
    raw = raw.strip()
    raw = sanitize_json_string(raw)

    # Try parsing JSON safely
    try:
        mapping = json.loads(raw)
        print(f"Group mapping: {mapping}")
    except Exception as e:
        print(f"Failed to parse regrouping JSON: {e}")
        print(f"Raw output:\n{raw}")
        return df

    # Remap groupings
    df['group'] = df['group'].map(lambda g: mapping.get(g, g))
    
    # Show final groups
    final_counts = df['group'].value_counts().to_dict()
    print(f"Final groups: {final_counts}")

    return df

if __name__ == "__main__":
    print("Loading raw data...")
    raw_data = load_data("raw_data")
    raw_data = raw_data.iloc[0:3]
    print(f"Loaded {len(raw_data)} posts")
    
    # For testing, uncomment this line to process only first 5 posts
    # raw_data = raw_data.iloc[0:5]
    
    print("Getting OpenRouter client...")
    client = get_openrouter_client()

    print("Processing data to extract advice...")
    processed_data = add_advice(client, raw_data)
    
    if len(processed_data) > 0:
        print("Regrouping similar categories...")
        regroup_clusters(client, processed_data)

        # print("Saving processed data...")
        # save_data(processed_data, "processed_data")
        # print(f"Successfully saved {len(processed_data)} pieces of advice")
    else:
        print("No advice was extracted. Check your input data and API responses.")
        
    print("Processing complete!")