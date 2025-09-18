import json
import re
import os
import time
from dotenv import load_dotenv
import pandas as pd
from openai import OpenAI
from database import *

# Configuration
FREE_OPENROUTER_MODEL = "openrouter/sonoma-dusk-alpha"
MAX_CHARS = 10000  # approximately 2500 tokens
BATCH_SIZE = 10  # Process posts in batches

load_dotenv()

def get_openrouter_client():
    """Get OpenRouter client with error handling"""
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    client = OpenAI(
        api_key=key,
        base_url="https://openrouter.ai/api/v1",
        timeout=90.0,
        max_retries=2,
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
    """Extract advice from text using AI"""
    doc_clean = doc.replace('"', "'").replace('\n', ' ').replace('\r', ' ')
    doc_clean = ' '.join(doc_clean.split())

    user = f"""
Goal: From TEXT, extract each independent piece of advice verbatim and assign it to a theme.

Return JSON only (no markdown/extras), exactly in this schema:
{{
    "items": [
        {{"advice": "example advice here", "group": "Theme Name"}}
    ]
}}

Rules:
- "advice": a verbatim substring of TEXT that is a self-contained actionable tip
- Replace all double quotes in advice text with single quotes
- "group": a short label (≤3 words) describing the theme of that advice
- Every "advice" must make complete sense on its own
- Use 2-6 distinct group labels overall
- If no clear advice found, return {{"items": []}}
- Remove formatting symbols like **bold**, *italics*, [links](url)

TEXT: {doc_clean[:MAX_CHARS]}
"""

    completion = prompt_ai(
        client, 
        messages=[
            {"role": "system", "content": "You are a careful extraction engine. Output must be strictly valid JSON. Start your response with '{' and end it with '}'."},
            {"role": "user", "content": user}
        ]
    )

    if not completion or not completion.choices:
        return {"items": []}

    raw = completion.choices[0].message.content
    if not raw:
        return {"items": []}
        
    raw = raw.strip()
    raw = sanitize_json_string(raw)

    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict) or "items" not in parsed:
            return {"items": []}
        return parsed
    except Exception as e:
        print(f"Failed to parse JSON: {e}")
        return {"items": []}

def process_posts_batch(client, posts_df):
    """
    Process a batch of posts to extract advice
    Returns DataFrame with advice extracted
    """
    if len(posts_df) == 0:
        return pd.DataFrame()

    print(f"Processing batch of {len(posts_df)} posts...")
    
    # Prepare text for processing
    posts_df = posts_df.copy()
    posts_df['alltext'] = get_alltext(posts_df)
    
    # Filter out posts with very little text
    posts_df = posts_df[posts_df['alltext'].str.len() >= 50].copy()
    
    if len(posts_df) == 0:
        print("No posts with sufficient text found")
        return pd.DataFrame()

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
            
            print(f"Post {row['id']}: Found {len(items)} advice items")
            
        except Exception as e:
            print(f"Error processing post {row['id']}: {e}")
            continue
    
    if not all_advice:
        print("No advice extracted from batch")
        return pd.DataFrame()
    
    advice_df = pd.DataFrame(all_advice)
    
    # Filter out empty advice
    advice_df = advice_df[advice_df['advice'].str.len() > 0].copy()
    
    print(f"Extracted {len(advice_df)} pieces of advice from batch")
    return advice_df

def regroup_advice_clusters(client, max_groups=25):
    """
    Regroup existing advice in database using AI
    """
    print("Loading current advice from database...")
    advice_df = get_all_advice()
    
    if len(advice_df) == 0:
        print("No advice found to regroup")
        return 0
    
    # Get current group counts
    group_counts = advice_df['group_name'].value_counts().to_dict()
    print(f"Current groups ({len(group_counts)}): {group_counts}")
    
    if len(group_counts) <= max_groups:
        print(f"Already have {len(group_counts)} groups, which is <= {max_groups}. No regrouping needed.")
        return 0
    
    # Build regrouping prompt
    user_prompt = f"""
You have {len(group_counts)} group labels that need consolidation to {max_groups} or fewer.

Current labels with counts: {group_counts}

Consolidate similar/synonymous groups into canonical categories.
Return JSON mapping each input label to its consolidated category:
{{"old_label": "new_canonical_label", ...}}

Rules:
- Merge very similar themes (e.g., "Sleep", "Sleep Schedule", "Sleep Habits" → "Sleep")  
- Keep labels short (1-3 words)
- Must reduce to {max_groups} or fewer distinct output values
- Be aggressive in merging to hit the target
"""

    completion = prompt_ai(
        client,
        messages=[
            {"role": "system", "content": "Return only valid JSON mapping old labels to new canonical labels."},
            {"role": "user", "content": user_prompt}
        ]
    )

    if not completion or not completion.choices:
        print("Failed to get regrouping response")
        return 0

    raw = completion.choices[0].message.content
    if not raw:
        print("Empty regrouping response")
        return 0
        
    raw = sanitize_json_string(raw)

    try:
        mapping = json.loads(raw)
        print(f"Regrouping mapping: {mapping}")
        
        # Apply the mapping to database
        updated_count = update_advice_groups(mapping)
        
        # Show results
        new_advice_df = get_all_advice()
        new_group_counts = new_advice_df['group_name'].value_counts().to_dict()
        print(f"Regrouping complete: {len(group_counts)} → {len(new_group_counts)} groups")
        print(f"Updated {updated_count} advice entries")
        print(f"New groups: {new_group_counts}")
        
        return updated_count
        
    except Exception as e:
        print(f"Failed to parse regrouping JSON: {e}")
        return 0

def filter_invalid_advice(client, batch_size=50):
    """
    Filter out entries that aren't actually advice using AI
    """
    print("Loading advice to filter...")
    advice_df = get_all_advice()
    
    if len(advice_df) == 0:
        print("No advice to filter")
        return 0
    
    print(f"Filtering {len(advice_df)} advice items...")
    total_invalidated = 0
    
    # Process in batches
    for i in range(0, len(advice_df), batch_size):
        batch = advice_df.iloc[i:i + batch_size]
        
        # Create list for the batch
        advice_list = []
        for idx, row in batch.iterrows():
            advice_list.append(f"{row['advice_id']}: {row['advice_text']}")
        
        advice_text = "\n".join(advice_list)
        
        user_prompt = f"""
Review these advice items and identify which ones are NOT actually actionable advice.

Return JSON with advice_ids that should be marked as invalid:
{{"invalid_ids": [123, 456, 789]}}

Mark as INVALID if the text is:
- Just a question or complaint
- Personal anecdote without actionable advice  
- Incomplete fragments
- Not actually giving advice/tips

Mark as VALID if it contains clear, actionable instructions someone could follow.

Advice items:
{advice_text}
"""
        
        completion = prompt_ai(
            client,
            messages=[
                {"role": "system", "content": "You filter out non-advice content. Return only valid JSON."},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        if not completion or not completion.choices:
            continue
        
        raw = completion.choices[0].message.content
        if not raw:
            continue
        
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
            continue
    
    print(f"Filtering complete: invalidated {total_invalidated} non-advice items")
    return total_invalidated

def process_unprocessed_posts(batch_size=BATCH_SIZE, max_batches=None):
    """
    Main processing function: get unprocessed posts and extract advice
    """
    print("Initializing database...")
    init_database()
    
    print("Getting OpenRouter client...")
    client = get_openrouter_client()
    
    processed_count = 0
    batch_num = 0
    
    while True:
        if max_batches and batch_num >= max_batches:
            print(f"Reached max batches limit ({max_batches})")
            break
            
        # Get next batch of unprocessed posts
        unprocessed_posts = get_unprocessed_posts(limit=batch_size)
        
        if len(unprocessed_posts) == 0:
            print("No more unprocessed posts found")
            break
        
        batch_num += 1
        post_ids = unprocessed_posts['id'].tolist()
        
        print(f"\n--- Batch {batch_num}: Processing {len(unprocessed_posts)} posts ---")
        
        try:
            # Mark as processing
            mark_posts_processing(post_ids)
            
            # Process the batch
            advice_df = process_posts_batch(client, unprocessed_posts)
            
            if len(advice_df) > 0:
                # Save advice to database
                saved_count = save_advice_to_db(advice_df)
                print(f"Saved {saved_count} advice items to database")
            
            # Mark as completed
            mark_posts_completed(post_ids)
            processed_count += len(unprocessed_posts)
            
            print(f"Batch {batch_num} completed successfully")
            
        except Exception as e:
            print(f"Error processing batch {batch_num}: {e}")
            mark_posts_failed(post_ids, str(e))
            continue
    
    print(f"\nProcessing complete! Processed {processed_count} posts in {batch_num} batches")
    return processed_count

def run_full_analysis_pipeline():
    """
    Run the complete analysis pipeline
    """
    print("=== STARTING FULL ANALYSIS PIPELINE ===\n")
    
    # Step 1: Process unprocessed posts
    print("Step 1: Processing unprocessed posts...")
    processed_count = process_unprocessed_posts()
    
    if processed_count == 0:
        print("No posts were processed, skipping remaining steps")
        return
    
    # Step 2: Regroup advice clusters
    print("\nStep 2: Regrouping advice clusters...")
    client = get_openrouter_client()
    regroup_advice_clusters(client, max_groups=25)
    
    # Step 3: Filter invalid advice
    print("\nStep 3: Filtering invalid advice...")
    filter_invalid_advice(client)
    
    # Step 4: Show final statistics
    print("\nStep 4: Final statistics...")
    show_analysis_stats()
    
    print("\n=== PIPELINE COMPLETE ===")

def show_analysis_stats():
    """Display comprehensive analysis statistics"""
    stats = get_processing_stats()
    advice_by_group = get_advice_by_group()
    
    print("\n=== ANALYSIS STATISTICS ===")
    
    print("\nProcessing Status:")
    for status, count in stats['processing_status'].items():
        print(f"  {status}: {count}")
    
    print("\nAdvice Stats:")
    for key, value in stats['advice_stats'].items():
        print(f"  {key}: {value}")
    
    print(f"\nTop 10 Advice Categories:")
    print(advice_by_group.head(10).to_string(index=False))
    
    print(f"\nAdvice per Category (showing top 15):")
    for _, row in advice_by_group.head(15).iterrows():
        print(f"  {row['group_name']}: {row['advice_count']} pieces from {row['post_count']} posts (avg {row['avg_upvotes']:.1f} upvotes)")

def export_advice_to_csv(filename="final_advice_export.csv"):
    """Export all valid advice to CSV for external use"""
    advice_df = get_all_advice()
    
    if len(advice_df) == 0:
        print("No advice to export")
        return
    
    # Clean and format for export
    export_df = advice_df[['advice_text', 'group_name', 'ups', 'title', 'subreddit_name_prefixed']].copy()
    export_df.columns = ['advice', 'category', 'upvotes', 'source_post_title', 'subreddit']
    
    # Sort by upvotes and category
    export_df = export_df.sort_values(['category', 'upvotes'], ascending=[True, False])
    
    export_df.to_csv(f"data/{filename}", index=False, encoding='utf-8')
    print(f"Exported {len(export_df)} advice items to data/{filename}")