from webscrape import *
from database import *

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

PROCESSING_VERSION = 1.1

# FREE_OPENROUTER_MODEL = "mistralai/devstral-small-2505:free" # Super lightweight model
# FREE_OPENROUTER_MODEL = "openrouter/sonoma-dusk-alpha"
FREE_OPENROUTER_MODEL = "x-ai/grok-4-fast:free"
FREE_OPENROUTER_MODEL = "deepseek/deepseek-chat-v3.1:free"

MAX_CHARS = 10000 # approximately 2500 tokens
BATCH_SIZE = 10

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

def get_alltext(df):
    """Combine title and body text from a dataframe"""
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
- "advice": a verbatim substring of TEXT that is a self-contained actionable tip. It may be as long as necessary to convey the point. It should make sense standalone. It should be quotable.
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

def process_posts_batch(client, posts_df):
    """Given a dataframe of posts to extract advice, returns a DataFrame with advice extracted"""
    posts_df = posts_df.copy()
    print(f"Processing batch of {len(posts_df)} posts...")

    # Clean and prepare the text
    posts_df['alltext'] = get_alltext(posts_df)

    # Extract advice from each post
    all_advice = []

    for idx, row in posts_df.iterrows():
        try:
            text = row['alltext'][:MAX_CHARS]
            parsed = gather_advice_and_group(client, text)
            items = parsed.get("items", [])
            
            # Create one row per advice item
            for item in items:
                if isinstance(item, dict) and 'advice' in item and 'group' in item:
                    advice_row = {
                        'id': row['id'],
                        'title': row['title'],
                        'subreddit_name_prefixed': row['subreddit_name_prefixed'],
                        'ups': row['ups'],
                        'upvote_ratio': row['upvote_ratio'],
                        'advice': item['advice'],
                        'group': item['group']
                    }
                    all_advice.append(advice_row)
                        
        except Exception as e:
            print(f"Error processing post {row['id']}: {e}")
            continue

    advice_df = pd.DataFrame(all_advice)

    print(f"Extracted {len(advice_df)} pieces of advice from batch")

    return advice_df

def regroup_clusters(client, max_groups=None):
    """Regroup existing advice in database using API"""
    print("Loading advice for regrouping...")
    advice_df = get_all_advice()
        
    # Get label counts to help the model choose canonical names
    counts = advice_df['group_name'].value_counts().to_dict()

    # ----- Prompt Engineering -----
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
        messages=[{"role": "user", "content": user_prompt}],
    )

    if not completion or not completion.choices:
        print("Failed to get regrouping response, keeping original groups")
        return 0

    raw = completion.choices[0].message.content
    if not raw:
        print("Empty regrouping response, keeping original groups")  
        return 0
        
    raw = raw.strip()
    raw = sanitize_json_string(raw)

    # Try parsing JSON safely
    try:
        mapping = json.loads(raw)

        unique_outputs = len(set(mapping.values()))
        print(f"Mapping creates {unique_outputs} unique output groups")

        if max_groups and unique_outputs > max_groups * 1.2:  # Allow 20% over target
            print(f"WARNING: AI didn't consolidate enough! Got {unique_outputs}, wanted {max_groups}")

        updated_count = update_advice_groups(mapping)

        new_advice_df = get_all_advice()
        new_counts = new_advice_df['group_name'].value_counts().to_dict()

        print(f"Regrouping complete: {len(counts)} → {len(new_counts)}")

        return updated_count # Number of rows updated

    except Exception as e:
        print(f"Failed to parse regrouping JSON: {e}")
        return 0

def filter_advice(client, batch_size=25):
    """
    Filter out rows that don't contain standalone advice
    """
    print("Loading advice to filter...")
    advice_df = get_all_advice()
    
    print(f"Starting filtering and formatting with {len(advice_df)} advice items")
    
    # Process advice in batches to avoid token limits
    total_invalidated = 0
    
    for i in range(0, len(advice_df), batch_size):
        batch = advice_df.iloc[i:i + batch_size]
        
        # Create list of advice items for the batch
        advice_list = []
        for idx, row in batch.iterrows():
            advice_list.append(f"{row['advice_id']}: {row['advice_text']}")
        
        advice_text = "\n".join(advice_list)
        
        user_prompt = f"""
Review these advice items and identify which ones are NOT actually actionable advice.

Return JSON with advice_ids that should be marked as invalid:
{{"invalid_ids": [123, 456, 789]}}

Mark as INVALID if the text is:
- Not actionable and specific
- Doesn't make sense standalone
- Doesn't contain clear advice or instruction or recommendation

INVALID Examples
- "Do the fucking thing"
- "pay that month's bill manually"
- "Turn off notifications"
- "I would 1000% suggest reading her book"
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
            continue
        
        raw = completion.choices[0].message.content
        if not raw:
            continue
        
        raw = raw.strip()
        raw = sanitize_json_string(raw)
        
        try:
            parsed = json.loads(raw)
            invalid_ids = parsed.get("invalid_ids", [])

            if invalid_ids:
                invalidated = invalidate_advice(invalid_ids)
                total_invalidated += invalidated
                print(f"Batch {i//batch_size + 1}: Invalidated {invalidated} items")

        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {e}")
    
    print(f"Filtering complete: invalidated {total_invalidated} non-advice items")

    return total_invalidated

def process_unprocessed_posts(client, batch_size=BATCH_SIZE):
    """
    Main processing function: get unprocessed posts and extract advice
    """
    init_database()
    
    processed_count = 0
    batch_num = 0
    
    while True:
        # Get next batch of unprocessed posts
        unprocessed_posts = get_unprocessed_posts(limit=batch_size)
        
        if len(unprocessed_posts) == 0:
            print("No more unprocessed posts found")
            break
        
        batch_num += 1
        post_ids = unprocessed_posts['id'].tolist()
        
        print(f"\n--- Batch {batch_num}: Processing {len(unprocessed_posts)} posts ---")
        
        try: 
            # Process the batch
            advice_df = process_posts_batch(client, unprocessed_posts)
            
            if len(advice_df) > 0:
                # Save advice to database
                saved_count = save_advice_to_db(advice_df)
                print(f"Saved {saved_count} advice items to database")
            
            # Mark as completed
            mark_posts_completed(post_ids, PROCESSING_VERSION, FREE_OPENROUTER_MODEL)
            processed_count += len(unprocessed_posts)
            
            print(f"Batch {batch_num} completed successfully")
            
        except Exception as e:
            print(f"Error processing batch {batch_num}: {e}")
            mark_posts_failed(post_ids, PROCESSING_VERSION, FREE_OPENROUTER_MODEL, str(e))
            continue
    
    print(f"\nProcessing complete! Processed {processed_count} posts in {batch_num} batches")
    return processed_count

def process_outdated_advice_posts(client, batch_size=BATCH_SIZE):
    """
    Reprocess posts that used an older processing version.
    """
    init_database()
    
    processed_count = 0
    batch_num = 0
    
    while True:
        # Get next batch of unprocessed posts
        outdated_posts = get_outdated_advice_posts(PROCESSING_VERSION, limit=batch_size)
        
        if len(outdated_posts) == 0:
            print("No outdated posts found")
            break
        
        batch_num += 1
        post_ids = outdated_posts['id'].tolist()
        
        print(f"\n--- Batch {batch_num}: Processing {len(outdated_posts)} posts ---")
        
        try: 
            # Process the batch
            advice_df = process_posts_batch(client, outdated_posts)
            
            if len(advice_df) > 0:
                # Save advice to database
                saved_count = save_advice_to_db(advice_df)
                print(f"Saved {saved_count} advice items to database")
            
            # Mark as completed
            mark_posts_completed(post_ids, PROCESSING_VERSION, FREE_OPENROUTER_MODEL)
            processed_count += len(outdated_posts)
            
            print(f"Batch {batch_num} completed successfully")
            
        except Exception as e:
            print(f"Error processing batch {batch_num}: {e}")
            mark_posts_failed(post_ids, PROCESSING_VERSION, FREE_OPENROUTER_MODEL, str(e))
            continue
    
    print(f"\nProcessing complete! Processed {processed_count} posts in {batch_num} batches")
    return processed_count