import sqlite3
import pandas as pd
import os
from contextlib import contextmanager

DATABASE_PATH = "data/advice_aggregator.db"

def init_database():
    """Initialize the database"""
    os.makedirs("data", exist_ok=True)
    
    with sqlite3.connect(DATABASE_PATH) as conn:
        conn.executescript('''
        -- Raw scraped posts
        CREATE TABLE IF NOT EXISTS posts (
            id TEXT PRIMARY KEY,
            title TEXT,
            selftext TEXT,
            subreddit_name_prefixed TEXT,
            ups INTEGER,
            upvote_ratio REAL,
            scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Extracted advice pieces
        CREATE TABLE IF NOT EXISTS advice (
            advice_id INTEGER PRIMARY KEY AUTOINCREMENT,
            post_id TEXT REFERENCES posts(id),
            advice_text TEXT,
            group_name TEXT,
            regroup_name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_duplicate BOOLEAN DEFAULT FALSE,
            is_safe BOOLEAN DEFAULT TRUE
        );

        -- Processing status tracking
        CREATE TABLE IF NOT EXISTS processing_status (
            post_id TEXT PRIMARY KEY REFERENCES posts(id),
            status TEXT, -- 'scraped', 'completed', 'failed'
            processing_version REAL,
            model_name TEXT,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            error_message TEXT
        );

        -- Advice score tracking
        CREATE TABLE IF NOT EXISTS advice_quality_scores (
            advice_id INTEGER PRIMARY KEY REFERENCES advice(advice_id),
            clarity REAL, 
            completeness REAL,
            practicality REAL,
            universality REAL,
            quality_score REAL,
            scored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Create indexes for better performance
        CREATE INDEX IF NOT EXISTS idx_advice_group ON advice(group_name);
        CREATE INDEX IF NOT EXISTS idx_advice_post ON advice(post_id);
        CREATE INDEX IF NOT EXISTS idx_status ON processing_status(status);
        ''')

@contextmanager
def get_db_connection():
    """Simple connection helper"""
    conn = sqlite3.connect(DATABASE_PATH)
    try:
        yield conn
    finally:
        conn.close()

def save_posts_to_db(posts_df):
    """Save posts dataframe to database, avoiding duplicates"""
    if len(posts_df) == 0:
        return 0
    
    with get_db_connection() as conn:
        # Get existing post IDs to avoid duplicates
        existing_ids = pd.read_sql(
            "SELECT id FROM posts", 
            conn
        )['id'].tolist()
        
        # Filter out existing posts
        new_posts = posts_df[~posts_df['id'].isin(existing_ids)].copy()
        
        if len(new_posts) == 0:
            print("No new posts to save")
            return 0
        
        # Save new posts
        new_posts.to_sql('posts', conn, if_exists='append', index=False)
        
        # Initialize processing status for new posts
        status_data = pd.DataFrame({
            'post_id': new_posts['id'],
            'status': 'scraped'
        })
        status_data.to_sql('processing_status', conn, if_exists='append', index=False)
        
        return len(new_posts)

def get_unprocessed_posts(limit=None):
    """Get posts that haven't been processed for advice yet"""
    query = '''
    SELECT p.* FROM posts p
    JOIN processing_status ps ON ps.post_id = p.id
    WHERE (
        ps.status IN ('scraped', 'failed')
    )
    ORDER BY p.ups DESC, p.scraped_at ASC
    '''

    #  OR ps.processing_version < ?
    
    if limit:
        query += f" LIMIT {limit}"
    
    with get_db_connection() as conn:
        return pd.read_sql(query, conn)

def get_outdated_advice_posts(processing_version, limit=None):
    """Get posts with advice processed by older model versions"""
    query = '''
    SELECT p.* FROM posts p
    JOIN processing_status ps ON ps.post_id = p.id
    WHERE ps.status = 'completed' AND ps.processing_version < ?
    ORDER BY p.ups DESC, p.scraped_at ASC
    '''

    if limit:
        query += f" LIMIT {limit}"
    
    with get_db_connection() as conn:
        return pd.read_sql(query, conn, params=(processing_version,))

def mark_posts_completed(post_ids, processing_version, model_name):
    """Mark posts as successfully processed"""
    if not post_ids:
        return
        
    with get_db_connection() as conn:
        placeholders = ','.join(['?' for _ in post_ids])
        params = [processing_version, model_name, *post_ids]
        
        conn.execute(f'''
        UPDATE processing_status 
        SET status = 'completed', 
            last_updated = CURRENT_TIMESTAMP, 
            processing_version = ?, 
            model_name = ?
        WHERE post_id IN ({placeholders})
        ''', params)
        conn.commit()


def mark_posts_failed(post_ids, processing_version, model_name, error_message="Processing failed"):
    """Mark posts as failed to process"""
    if not post_ids:
        return
        
    with get_db_connection() as conn:
        for post_id in post_ids:
            conn.execute('''
            UPDATE processing_status 
            SET status = 'failed', 
                error_message = ?, 
                last_updated = CURRENT_TIMESTAMP, 
                processing_version = ?, 
                model_name = ?
            WHERE post_id = ?
            ''', (error_message, processing_version, model_name, post_id))
        conn.commit()

def save_advice_to_db(advice_df):
    """Save extracted advice to database, replacing any existing advice for these posts"""
    if len(advice_df) == 0:
        return 0
        
    # Prepare advice data for insertion
    advice_data = advice_df[['id', 'advice', 'group']].copy()
    advice_data.columns = ['post_id', 'advice_text', 'group_name']

    # Add regroup name as a copy of the group_name
    advice_data["regroup_name"] = advice_data["group_name"]
    
    with get_db_connection() as conn:
        # Get unique post IDs that we're about to process
        post_ids = advice_data['post_id'].unique().tolist()
        
        # Delete existing advice for these posts to avoid duplicates
        if post_ids:
            placeholders = ','.join(['?' for _ in post_ids])
            deleted = conn.execute(f'''
                DELETE FROM advice 
                WHERE post_id IN ({placeholders})
            ''', post_ids)
            
            if deleted.rowcount > 0:
                print(f"Removed {deleted.rowcount} existing advice entries for reprocessing")
        
        # Insert new advice
        advice_data.to_sql('advice', conn, if_exists='append', index=False)
        conn.commit()
        
        return len(advice_data)

def save_quality_scores_to_db(scores_df):
    """Save quality scores to database"""
    if len(scores_df) == 0:
        return
    
    with get_db_connection() as conn:
        # Clear existing scores for these advice items
        advice_ids = scores_df['advice_id'].tolist()
        if advice_ids:
            placeholders = ','.join(['?' for _ in advice_ids])
            conn.execute(f'DELETE FROM advice_quality_scores WHERE advice_id IN ({placeholders})', advice_ids)
        
        # Insert new scores
        scores_df.to_sql('advice_quality_scores', conn, if_exists='append', index=False)
        conn.commit()

def update_advice_groups(group_mapping):
    """Update advice (re)group names based on mapping"""
    if not group_mapping:
        return 0
        
    updated_count = 0
    with get_db_connection() as conn:
        for old_group, new_group in group_mapping.items():
            result = conn.execute('''
            UPDATE advice 
            SET regroup_name = ? 
            WHERE regroup_name = ?
            ''', (new_group, old_group))
            updated_count += result.rowcount
        conn.commit()
    
    return updated_count

def get_table_columns(table_name):
    with get_db_connection() as conn:
        cursor = conn.execute(f"PRAGMA table_info({table_name})")
        return [row[1] for row in cursor.fetchall()]

def mark_advice_flag(advice_ids, flag_name, value=True):
    """Mark a boolean-style flag column on advice rows."""
    if not advice_ids:
        return 0

    # Dynamically get column names from the advice table
    valid_columns = get_table_columns("advice")
    if flag_name not in valid_columns:
        raise ValueError(f"Invalid flag name: {flag_name}")

    with get_db_connection() as conn:
        placeholders = ",".join(["?" for _ in advice_ids])
        sql = f"UPDATE advice SET {flag_name} = ? WHERE advice_id IN ({placeholders})"
        result = conn.execute(sql, [int(value)] + advice_ids)
        conn.commit()
        return result.rowcount

def cleanup_failed_processing():
    """Reset failed posts back to scraped status for retry"""
    with get_db_connection() as conn:
        result = conn.execute('''
        UPDATE processing_status 
        SET status = 'scraped', error_message = NULL, last_updated = CURRENT_TIMESTAMP
        WHERE status = 'failed'
        ''')
        conn.commit()
        return result.rowcount

def get_unscored_advice():
    """Get all valid advice that doesn't have quality scores yet"""
    with get_db_connection() as conn:
        return pd.read_sql('''
        SELECT 
            a.advice_id,
            a.post_id,
            a.advice_text,
            a.group_name,
            a.regroup_name,
            a.created_at,
            p.title,
            p.ups,
            p.upvote_ratio,
            p.subreddit_name_prefixed
        FROM advice a
        JOIN posts p ON a.post_id = p.id
        LEFT JOIN advice_quality_scores q ON a.advice_id = q.advice_id
        WHERE a.is_duplicate = FALSE AND a.is_safe = TRUE AND q.universality IS NULL
        ORDER BY p.ups DESC, a.created_at DESC
        ''', conn)

def get_all_advice():
    """Get all valid advice."""
    with get_db_connection() as conn:
        return pd.read_sql('''
        SELECT 
            a.advice_id,
            a.post_id,
            a.advice_text,
            a.group_name,
            a.regroup_name,
            a.created_at,
            p.title,
            p.ups,
            p.upvote_ratio,
            p.subreddit_name_prefixed
        FROM advice a
        JOIN posts p ON a.post_id = p.id
        WHERE a.is_duplicate = FALSE AND a.is_safe = TRUE
        ORDER BY p.ups DESC, a.created_at DESC
        ''', conn)

def get_full_database():
    """
    Load the entire database into a single merged DataFrame.
    Joins advice with posts, processing_status, and advice_quality_scores.
    """
    with sqlite3.connect(DATABASE_PATH) as conn:
        # Load each table into a DataFrame
        posts_df = pd.read_sql("SELECT * FROM posts", conn)
        advice_df = pd.read_sql("SELECT * FROM advice", conn)
        status_df = pd.read_sql("SELECT * FROM processing_status", conn)
        scores_df = pd.read_sql("SELECT * FROM advice_quality_scores", conn)

    # Merge advice with posts
    merged = advice_df.merge(posts_df, how="left", left_on="post_id", right_on="id", suffixes=("_advice", "_post"))

    # Merge with processing_status
    merged = merged.merge(status_df, how="left", left_on="post_id", right_on="post_id", suffixes=("", "_status"))

    # Merge with quality scores
    merged = merged.merge(scores_df, how="left", left_on="advice_id", right_on="advice_id", suffixes=("", "_score"))

    return merged


def export_advice_to_csv(filename, quality_threshold=0.0):
    """Export advice in a clean format optimized for website display"""
    with get_db_connection() as conn:
        df = pd.read_sql('''
        SELECT 
            a.advice_text as advice,
            a.group_name as old_category,
            a.regroup_name as category,
            p.id as id,
            p.ups as upvotes,
            p.upvote_ratio as upvote_ratio,
            p.subreddit_name_prefixed as subreddit,
            s.clarity as clarity_score,
            s.completeness as completeness_score,
            s.practicality as practicality_score,
            s.universality as universality_score,
            s.quality_score as quality_score
        FROM advice a
        JOIN posts p ON a.post_id = p.id
        JOIN advice_quality_scores s ON s.advice_id = a.advice_id
        WHERE a.is_duplicate = FALSE 
            AND a.is_safe = TRUE
            AND s.quality_score >= ?
            AND a.regroup_name <> ?
        ORDER BY a.regroup_name, p.ups DESC
        ''', conn, params=[quality_threshold, "placeholder"])
    
    if len(df) == 0:
        print("No advice to export")
        return
    
    # Clean for website display
    df['advice'] = df['advice'].str.strip()
    df['category'] = df['category'].str.title()
    
    # Export
    df.to_csv(f"data/{filename}", index=False, encoding='utf-8')