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

# FREE_OPENROUTER_MODEL = "mistralai/devstral-small-2505:free" # Super lightweight model
FREE_OPENROUTER_MODEL = "openrouter/sonoma-dusk-alpha"
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
- "advice": a verbatim substring of TEXT that is a self-contained actionable tip. It may be as long as necessary to convey the point.
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

def regroup_clusters(client, df, max_groups=None):
    """Given a dataframe, modifies the data inplace and regroups the groupings"""
    if len(df) == 0:
        print("No data to regroup")
        return df
        
    # Get label counts to help the model choose canonical names
    counts = df['group'].value_counts().to_dict()
    print(f"Original groups: {counts}")

    constraint = (
        f"\nIMPORTANT: Ensure the total number of DISTINCT output values is at most {max_groups}. "
        f"You currently have {len(counts)} groups and need to reduce to {max_groups} or fewer."
        if max_groups else ""
    )

    # Build a small prompt
    user_prompt = f"""
You are given a list of existing group labels with their frequencies.
Return a JSON object that maps EACH old label to ONE consolidated canonical label.
Keep labels short (1-3 words). Merge synonyms/redundant labels into a single canonical label.
Output JSON only, no markdown or commentary.{constraint}

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

def regroup_clusters(client, df, max_groups=None):
    """
    Improved version with better debugging and error handling
    """
    if len(df) == 0:
        print("No data to regroup")
        return df
    
    # Make a copy to avoid modifying original
    df = df.copy()
        
    # Get label counts to help the model choose canonical names
    counts = df['group'].value_counts().to_dict()
    print(f"Original groups ({len(counts)} total): {counts}")

    # If we already have few enough groups, skip regrouping
    if max_groups and len(counts) <= max_groups:
        print(f"Already have {len(counts)} groups, which is <= {max_groups}. Skipping regrouping.")
        return df
    
    # Base examples (always included)
    examples = """
Examples of required consolidation from your data:
- "Habits", "Daily Habits", "Healthy Habits", "Habit Tracking" → ALL become "Habits"
- "Self-Care", "Self-Love", "Self-Acceptance", "Self-Compassion" → ALL become "Self-Care" 
- "Focus", "Focus Management", "Focus Duration", "Focus Enjoyment" → ALL become "Focus"
- "Time Management", "Task Management", "Planning", "Organization" → ALL become "Planning"
"""
    
    # Conditional constraint
    if max_groups:
        constraint = f"""
CRITICAL TARGET: You must reduce {len(counts)} groups down to {max_groups} or fewer groups.
This means MANY input labels must map to the SAME output labels. Be very aggressive in merging.
"""
    else:
        constraint = """
GOAL: Consolidate similar/synonymous groups into canonical categories to reduce redundancy.
"""
    
    # Build prompt
    user_prompt = f"""
You have {len(counts)} group labels that need consolidation.

{constraint}

{examples}

Current labels: {list(counts.keys())}

Return JSON mapping each input to its consolidated category:
{{"input_label": "canonical_category", ...}}
"""

    # Call the LLM
    completion = prompt_ai(
        client,
        messages=[
            {"role": "system", "content": "You are a label consolidation expert. Return only valid JSON mapping old labels to new canonical labels."},
            {"role": "user", "content": user_prompt}
        ],
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
    print(f"Raw API response: {raw[:200]}...")  # Debug print

    # Try parsing JSON safely
    try:
        mapping = json.loads(raw)
        print(f"Parsed mapping: {mapping}")
        
        # Validate that all original groups are in the mapping
        missing_keys = set(counts.keys()) - set(mapping.keys())
        if missing_keys:
            print(f"Warning: Missing mappings for groups: {missing_keys}")
            # Add identity mappings for missing keys
            for key in missing_keys:
                mapping[key] = key

        # Count unique output values
        unique_outputs = len(set(mapping.values()))
        print(f"Mapping creates {unique_outputs} unique output groups")

        if max_groups and unique_outputs > max_groups * 1.2:  # Allow 20% over target
            print(f"WARNING: AI didn't consolidate enough! Got {unique_outputs}, wanted {max_groups}")
            # You could add fallback logic here
                
    except Exception as e:
        print(f"Failed to parse regrouping JSON: {e}")
        print(f"Raw output:\n{raw}")
        return df

    # Apply the mapping
    original_groups = df['group'].value_counts()
    df['group'] = df['group'].map(lambda g: mapping.get(g, g))
    
    # Show results
    final_counts = df['group'].value_counts().to_dict()
    print(f"Final groups ({len(final_counts)} total): {final_counts}")
    print(f"Reduced from {len(original_groups)} to {len(final_counts)} groups")

    return df

def filter_and_format_advice(client, df):
    """
    Filter out rows that don't contain standalone advice and fix formatting/capitalization
    """
    if len(df) == 0:
        print("No data to filter and format")
        return df
    
    print(f"Starting filtering and formatting with {len(df)} advice items")
    
    # Process advice in batches to avoid token limits
    batch_size = 20
    filtered_items = []
    
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i + batch_size]
        
        # Create list of advice items for the batch
        advice_list = []
        for idx, row in batch.iterrows():
            advice_list.append(f"{idx}: {row['advice']}")
        
        advice_text = "\n".join(advice_list)
        
        user_prompt = f"""
You are given a list of extracted advice items. For each item:
1. Decide if it's a valid, standalone piece of advice (actionable tip someone could follow)
2. If valid, improve the formatting and capitalization to make it clear and professional

Return JSON only (no markdown), in this exact schema:
{{
    "items": [
        {{"index": original_index, "advice": "improved advice text", "keep": true}},
        {{"index": original_index, "advice": "", "keep": false}}
    ]
}}

Rules for KEEP=TRUE (valid advice):
- Must be actionable and specific
- Should make sense as standalone advice
- Contains clear instructions or recommendations
- Not just questions, complaints, or personal anecdotes

Rules for FORMATTING (when keep=true):
- Fix capitalization and punctuation
- Make sentences complete and clear
- Remove redundant words or awkward phrasing
- Keep the core meaning intact
- Start with capital letter, end with period if sentence
- Replace single quotes with double quotes if needed

INVALID examples (keep=false):
- "I don't know what to do"
- "Anyone else have this problem?"
- "This happened to me"
- Incomplete fragments that don't give advice

VALID examples (keep=true, with formatting):
- "Set a consistent bedtime routine to improve sleep quality."
- "Try the 5-minute rule: commit to doing a task for just 5 minutes."

Advice items to process:
{advice_text}
"""
        
        completion = prompt_ai(
            client,
            messages=[
                {"role": "system", "content": "You are a careful content filter and formatter. Output must be strictly valid JSON. Start response with '{' and end with '}'."},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        if not completion or not completion.choices:
            print(f"No completion for batch {i//batch_size + 1}, keeping original")
            for idx, row in batch.iterrows():
                filtered_items.append({
                    "index": idx,
                    "advice": row['advice'],
                    "keep": True  # Default to keeping if API fails
                })
            continue
        
        raw = completion.choices[0].message.content
        if not raw:
            print(f"Empty response for batch {i//batch_size + 1}, keeping original")
            for idx, row in batch.iterrows():
                filtered_items.append({
                    "index": idx,
                    "advice": row['advice'],
                    "keep": True
                })
            continue
        
        raw = raw.strip()
        raw = sanitize_json_string(raw)
        
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and "items" in parsed:
                filtered_items.extend(parsed["items"])
            else:
                print(f"Invalid JSON structure for batch {i//batch_size + 1}, keeping original")
                for idx, row in batch.iterrows():
                    filtered_items.append({
                        "index": idx,
                        "advice": row['advice'],
                        "keep": True
                    })
        except Exception as e:
            print(f"Failed to parse JSON for batch {i//batch_size + 1}: {e}")
            # Keep original on parse failure
            for idx, row in batch.iterrows():
                filtered_items.append({
                    "index": idx,
                    "advice": row['advice'],
                    "keep": True
                })
    
    # Create mapping from original index to processed result
    result_map = {item["index"]: item for item in filtered_items}
    
    # Apply filtering and formatting
    keep_mask = []
    formatted_advice = []
    
    for idx, row in df.iterrows():
        if idx in result_map:
            result = result_map[idx]
            keep = result.get("keep", True)
            advice = result.get("advice", row['advice'])
        else:
            # Fallback if index not found
            keep = True
            advice = row['advice']
        
        keep_mask.append(keep)
        formatted_advice.append(advice)
    
    # Filter out rows marked as not advice
    df_filtered = df[keep_mask].copy()
    df_filtered['advice'] = [advice for advice, keep in zip(formatted_advice, keep_mask) if keep]
    
    kept_count = len(df_filtered)
    removed_count = len(df) - kept_count
    
    print(f"Filtering complete: kept {kept_count} advice items, removed {removed_count}")
    
    return df_filtered

def second_processing(data):
    """
    Second processing step: regroup clusters, filter non-advice, and format advice
    """
    if len(data) == 0:
        print("No data for second processing")
        return data
        
    client = get_openrouter_client()
    
    print("Regrouping clusters with max 30 groups...")
    regroup_clusters(client, data, max_groups=30)
    
    print("Filtering and formatting advice...")
    data_filtered = filter_and_format_advice(client, data)
    
    print(f"Second processing complete: {len(data_filtered)} final advice items")
    return data_filtered

def process_raw_data_and_save():
  print("Loading raw data...")
  raw_data = load_data("raw_data")
  print(f"Loaded {len(raw_data)} posts")
  
  print("Getting OpenRouter client...")
  client = get_openrouter_client()

  print("Processing data to extract advice...")
  processed_data = add_advice(client, raw_data)
  
  if len(processed_data) > 0:
      print("Regrouping similar categories...")
      regroup_clusters(client, processed_data)

      print("Saving processed data...")
      save_data(processed_data, "processed_data")
      print(f"Successfully saved {len(processed_data)} pieces of advice")
  else:
      print("No advice was extracted. Check your input data and API responses.")
      
  print("Processing complete!")

if __name__ == "__main__":
  # data = load_data("processed_data")
  # data = second_processing(data)

  # save_data(data, "p2")

  data = load_data("display_final_data")

  data = regroup_clusters(get_openrouter_client(), data, max_groups=25)

  save_data(data, "p3")