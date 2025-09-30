from webscrape import *
from database import *

import json
import re
import os
import statistics
import time

from dotenv import load_dotenv
import hdbscan
import numpy as np
from openai import OpenAI
import pandas as pd
from better_profanity import profanity
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
import umap

PROCESSING_VERSION = 1.1

# FREE_OPENROUTER_MODEL = "mistralai/devstral-small-2505:free" # Super lightweight model
# FREE_OPENROUTER_MODEL = "openrouter/sonoma-dusk-alpha"
# FREE_OPENROUTER_MODEL = "x-ai/grok-4-fast:free"
FREE_OPENROUTER_MODEL = "deepseek/deepseek-chat-v3.1:free"

MAX_CHARS = 10000 # approximately 2500 tokens
BATCH_SIZE = 10

load_dotenv()

# Remove words from the list of profanity
profanity.load_censor_words()
for word in ["stupid", "poop", "god", "kill", "suck", "piss", "potty"]:
    if word in profanity.CENSOR_WORDSET:
        profanity.CENSOR_WORDSET.remove(word)

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

def get_sentence_transformer():
    return SentenceTransformer('all-MiniLM-L6-v2')

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

def regroup_clusters_batch(client, batch_size, target_groups):
    """Regroup a batch of categories."""
    advice_df = get_all_advice()
    
    # Get smallest groups first (ascending=True) up to batch_size
    group_counts = advice_df['regroup_name'].value_counts(ascending=True)[:batch_size].to_dict()
    
    if not group_counts:
        print("No groups to process")
        return 0

    print(f"Processing batch of {len(group_counts)} groups")

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
    if target_groups:
        constraint = f"""
CRITICAL TARGET: You must reduce {len(group_counts)} groups down to {target_groups} or fewer groups.
This means MANY input labels must map to the SAME output labels. Be very aggressive in merging.
"""
    else:
        constraint = """
GOAL: Consolidate similar/synonymous groups into canonical categories to reduce redundancy.
"""
    
    # Build prompt
    user_prompt = f"""
You have {len(group_counts)} group labels that need consolidation.

{constraint}

{examples}

Current labels: {list(group_counts.keys())}

Return JSON mapping each input to its consolidated category:
{{"input_label": "canonical_category", ...}}
"""

    # Call the LLM
    completion = prompt_ai(
        client,
        messages=[{"role": "user", "content": user_prompt}],
    )

    if not completion or not completion.choices or not completion.choices[0].message.content:
        print("Failed to get AI response")
        return 0
        
    raw_response = completion.choices[0].message.content.strip()
    sanitized_response = sanitize_json_string(raw_response)

    # Try parsing JSON safely
    try:
        mapping = json.loads(sanitized_response)

        unique_outputs = len(set(mapping.values()))
        # print(f"Mapping creates {unique_outputs} unique output groups")

        if target_groups and unique_outputs > target_groups * 1.2:  # Allow 20% over target
            print(f"WARNING: AI didn't consolidate enough! Got {unique_outputs}, wanted {target_groups}")

        updated_count = update_advice_groups(mapping)

        new_advice_df = get_all_advice()
        new_counts = new_advice_df['regroup_name'].value_counts().to_dict()

        return updated_count # Number of rows updated

    except Exception as e:
        print(f"Failed to parse regrouping JSON: {e}")
        return 0
    
def regroup_clusters(client, batch_size=200, target_groups=None):
    """
    Regroup existing advice in database using API.
    Takes advice groupings in batches, smallest groups first, and regroups them together
    until there are fewer than max_groups groups. Then, if target_groups is specified, performs
    another round of grouping with a constraint.
    """

    advice_df = get_all_advice()
    counts = advice_df['regroup_name'].value_counts().to_dict()

    fail_max = 3
    updated_count = 0

    initial_group_count = len(counts)

    print(f"Starting with {initial_group_count} groups")
    print(f"Target: Reduce to ≤{batch_size} groups, then optionally to {target_groups}")

    i = 0 # number of batches so far
    f = 0 # number of failed batches so far
    while len(counts) > batch_size and f < fail_max:
        i += 1
        # Regroup clusters, then load clusters again
        pregroup_outputs = len(counts)

        updated = regroup_clusters_batch(client, batch_size, target_groups=None)

        if updated:
            advice_df = get_all_advice()
            counts = advice_df['regroup_name'].value_counts().to_dict()

            postgroup_outputs = len(counts)
            print(f"Regrouping batch {i} complete: {pregroup_outputs} → {postgroup_outputs}")

            updated_count += updated
        else:
            print(f"Failed to regroup batch {i}; failure number {f}")
            f += 1
    
    if f >= 3:
        print(f"Exceeded failure limit of {fail_max}")
        return updated_count
    
    # Regroup one final time, using a constraint if it exists
    if target_groups:
        print(f"Final Consolidation to ~{target_groups} groups")
    else:
        print(f"Final Consolidation")
    final_updated = regroup_clusters_batch(client, batch_size, target_groups=target_groups)
    updated_count += final_updated if final_updated else 0

    advice_df = get_all_advice()
    counts = advice_df['regroup_name'].value_counts().to_dict()

    print(f"Regrouping complete! {initial_group_count} → {len(counts)}")
    return updated_count

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

def filter_duplicates(model, similarity_threshold=0.90):
    """
    Use sentence embeddings to find and invalidate duplicate advice.
    From my testing, 0.9 seemed to be a good number.
    """
    print("Loading advice to filter...")
    advice_df = get_all_advice()
    total_removed = 0

    if len(advice_df) == 0:
        print("No advice found to filter")
        return 0

    print(f"Filtering duplicates from {len(advice_df)} advice items...")

    for category in advice_df["regroup_name"].unique():
        category_advice = advice_df[advice_df['regroup_name'] == category]

        if len(category_advice) < 10:
            print(f"Skipped {category} with only {len(category_advice)} advice items")
            continue
    
        print(f"Processing '{category}' ({len(category_advice)} items)")

        # Get embeddings for all advice in this category
        texts = category_advice['advice_text'].tolist()
        embeddings = model.encode(texts)

        similarity_matrix = cosine_similarity(embeddings)

        remove_indices = set()
        kept_indices = set()
        for i in range(len(similarity_matrix)):
            if i in remove_indices or i in kept_indices:
                continue
                
            # Find similar items
            similar_indices = np.where(similarity_matrix[i] >= similarity_threshold)[0]
            similar_indices = similar_indices[similar_indices != i]  # Exclude self
            
            if len(similar_indices) > 1:  # More than just self
                # Keep best one based on quality score, remove others
                candidates = category_advice.iloc[similar_indices]
                print(candidates['advice_text'].tolist())
                
                # Keep the one with the highest upvote ratio
                best_idx = candidates['upvote_ratio'].idxmax()
                kept_indices.add(best_idx)
                
                # Mark others for removal
                for idx in similar_indices:
                    if category_advice.iloc[idx].name != best_idx:
                        remove_indices.add(idx)
        
        # Remove duplicates
        if remove_indices:
            advice_ids_to_remove = category_advice.iloc[list(remove_indices)]['advice_id'].tolist()
            removed = mark_advice_flag(advice_ids_to_remove, "is_duplicate", True)
            total_removed += removed
            print(f"  Removed {removed} duplicates out of {len(advice_df)} items")
    
    print(f"Embedding deduplication complete: removed {total_removed} duplicates")
    return total_removed

def filter_links_and_profanity(batch_size=5000):
    """
    Filter links and profanities from advice text by invalidating them.
    """
    print("Loading advice to filter...")
    advice_df = get_all_advice()
    
    if len(advice_df) == 0:
        print("No advice found to filter")
        return 0
    
    print(f"Applying safety filters to {len(advice_df)} advice items...")
    total_unsafe_ids = 0
    
    # Process in batches
    for i in range(0, len(advice_df), batch_size):
        batch = advice_df.iloc[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(advice_df)-1)//batch_size + 1} ({len(batch)} items)")
        
        batch_unsafe_ids = []
        
        for _, row in batch.iterrows():
            advice_text = row['advice_text']
            
            # Filter links
            url_patterns = [
                r'https?://\S+',
                r'www\.\S+',
                r'\S+\.(com|org|net|edu|gov|io|co)\S*',
                r'discord\.gg/',
                r'bit\.ly/',
                r'tinyurl\.',
            ]
            
            if any(re.search(pattern, advice_text, re.IGNORECASE) for pattern in url_patterns):
                batch_unsafe_ids.append(row['advice_id'])
                # print(advice_text)
                continue
                
            # Profanity check
            if profanity.contains_profanity(advice_text):
                # print(advice_text)
                batch_unsafe_ids.append(row['advice_id'])
                continue
        
        # Mark unsafe items in this batch
        if batch_unsafe_ids:
            mark_advice_flag(batch_unsafe_ids, "is_safe", False)
            print(f"  Found and marked {len(batch_unsafe_ids)} unsafe items in this batch")
            total_unsafe_ids += len(batch_unsafe_ids)
        else:
            print(f"  No unsafe items found in this batch")
    
    print(f"Safety filtering complete: updated {total_unsafe_ids} items total")
    return total_unsafe_ids

def filter_sarcastic_or_harmful_advice(client, batch_size=20):
    """
    Use an LLM to flag advice that is sarcastic or harmful and mark it INVALID.
    """
    print("Loading advice to scan for sarcasm/harm...")
    advice_df = get_all_advice()

    if len(advice_df) == 0:
        print("No advice to filter")
        return 0

    print(f"Scanning {len(advice_df)} advice items...")
    total_invalidated = 0

    for i in range(0, len(advice_df), batch_size):
        batch = advice_df.iloc[i:i + batch_size]

        # Prepare "id: text" lines for this batch
        lines = []
        for _, row in batch.iterrows():
            lines.append(f"{row['advice_id']}: {row['advice_text']}")
        advice_block = "\n".join(lines)

        user_prompt = f"""
You are a safety reviewer. Identify advice that is clearly sarcastic or harmful.

Return ONLY valid JSON:
{{"invalid_ids": [123, 456]}}

Definitions:
- Sarcastic: mockery/irony that undermines clarity or ridicules.
- Harmful: promotes self-harm, harm to others, illegal acts, hate/harassment, dangerous medical/financial advice, or abusive language.

Mark INVALID if:
- Sarcastic to the point of being dismissive,
- Contains slurs, threats, harassment,
- Suggests dangerous/illegal actions.

Mark VALID if:
- Neutral, direct, or blunt but not attacking,
- Critiques ideas/actions without encouraging harm.

Review these items ("advice_id: advice_text"):
{advice_block}
"""

        completion = prompt_ai(
            client,
            messages=[
                {"role": "system", "content": "You filter out sarcastic or harmful advice. Return only valid JSON."},
                {"role": "user", "content": user_prompt}
            ]
        )

        if not completion or not getattr(completion, "choices", None):
            continue

        raw = completion.choices[0].message.content or ""
        raw = sanitize_json_string(raw)

        try:
            parsed = json.loads(raw)
            invalid_ids = parsed.get("invalid_ids", [])
            if invalid_ids:
                # mark these as invalid in DB
                invalidated = mark_advice_flag(invalid_ids, "is_safe", False)
                total_invalidated += invalidated
                print(f"Batch {i//batch_size + 1}: Invalidated {invalidated} sarcastic/harmful items")
                # for invalid_id in invalid_ids:
                #     print(advice_df[advice_df['advice_id'] == invalid_id]['advice_text'].tolist())
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {e}")
            continue

    print(f"Sarcasm/harm filtering complete: invalidated {total_invalidated} items")
    return total_invalidated


def harmonic_mean(scores):
    """Calculate harmonic mean, handling zero values"""
    valid_scores = [s for s in scores if s > 0]
    if not valid_scores:
        return 0.0

    harmonic_mean = statistics.harmonic_mean(valid_scores)

    return round(harmonic_mean, 2) # Truncate

def score_advice_quality(client, rescore_scored_advice=False, batch_size=20):
    """Score all advice using LLM with metrics"""

    # Old metrics:
    # - COMPLETENESS: Is the advice complete and standalone? 

    print("Loading advice for quality scoring...")
    if rescore_scored_advice:
        advice_df = get_all_advice()
    else:
        advice_df = get_unscored_advice()
    
    if len(advice_df) == 0:
        print("No advice found")
        return
    
    print(f"Scoring {len(advice_df)} advice items...")
    
    # Process in batches to avoid token limits
    total_scored = 0
    
    for i in range(0, len(advice_df), batch_size):
        batch = advice_df.iloc[i:i + batch_size]
        
        # Create advice items for scoring
        advice_items = []
        for _, row in batch.iterrows():
            advice_items.append({
                'id': row['advice_id'],
                'category': row['regroup_name'],
                'advice': row['advice_text']
            })
        
        # Build prompt
        items_text = ""
        for item in advice_items:
            items_text += f"ID: {item['id']}\nCategory: {item['category']}\nAdvice: {item['advice']}\n\n"
        
        prompt = f"""Score each piece of advice on 3 metrics from 0.0 to 1.0 (inclusive). 0.0 is very poor, 1.0 is excellent. Only use scores that are multiples of 0.1: 

CLARITY: How clear to average person?
COMPLETENESS: Stands alone and makes sense?
PRACTICALITY: Realistic actionable steps?
UNIVERSALITY: Broadly applicable and objectively good?

Examples:
- "Be happy" → 0.9, 0.1, 0.1, 0.2
- "Appoint someone you trust with passwords and give them physical copy" → 0.9, 0.9, 0.8, 0.9

Return JSON only:
{{
  "scores": [
    {{"id": 123, "clarity": 0.8, "completeness": 0.9, "practicality": 0.7, "universality": 0.7}},
    {{"id": 124, "clarity": 0.3, "completeness": 0.6, "practicality": 0.9, "universality": 0.4}}
  ]
}}

Advice to score:
{items_text}"""

        completion = prompt_ai(client, messages=[
            {"role": "system", "content": "You are a precise advice quality evaluator. Return only valid JSON with numeric scores."},
            {"role": "user", "content": prompt}
        ])
        
        if not completion or not completion.choices:
            print(f"  Batch {i//batch_size + 1} failed - no response")
            continue
        
        raw = sanitize_json_string(completion.choices[0].message.content)
        
        try:
            result = json.loads(raw)
            batch_scores = result.get("scores", [])
            
            # Calculate harmonic means and store results
            scored_advice = []
            for score_data in batch_scores:
                advice_id = score_data.get("id")
                clarity = score_data.get("clarity", 0) 
                completeness = score_data.get("completeness", 0)
                practicality = score_data.get("practicality", 0)
                universality = score_data.get("universality", 0)
                
                # Calculate harmonic mean
                quality_score = harmonic_mean([clarity, completeness, practicality, universality])
                
                scored_advice.append({
                    'advice_id': advice_id,
                    'clarity': clarity,
                    'completeness': completeness,
                    'practicality': practicality,
                    'universality': universality,
                    'quality_score': quality_score
                })
            
            if scored_advice:
                scores_df = pd.DataFrame(scored_advice)
                save_quality_scores_to_db(scores_df)
                total_scored += len(scored_advice)
                print(f"  Batch {i//batch_size + 1}: scored and saved {len(scored_advice)} items (total: {total_scored})")
            
        except Exception as e:
            print(f"  Batch {i//batch_size + 1} failed: {e}")
            continue
    
    if total_scored == 0:
        print("No advice was successfully scored")
        return None
    
    print(f"Quality scoring complete: {len(scored_advice)} items scored")
    
    return scores_df