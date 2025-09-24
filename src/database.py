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
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_valid BOOLEAN DEFAULT TRUE
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
    
def update_advice_groups(group_mapping):
    """Update advice group names based on mapping"""
    if not group_mapping:
        return 0
        
    updated_count = 0
    with get_db_connection() as conn:
        for old_group, new_group in group_mapping.items():
            result = conn.execute('''
            UPDATE advice 
            SET group_name = ? 
            WHERE group_name = ?
            ''', (new_group, old_group))
            updated_count += result.rowcount
        conn.commit()
    
    return updated_count

def invalidate_advice(advice_ids):
    """Mark advice as invalid (not real advice)"""
    if not advice_ids:
        return 0
        
    with get_db_connection() as conn:
        placeholders = ','.join(['?' for _ in advice_ids])
        result = conn.execute(f'''
        UPDATE advice 
        SET is_valid = FALSE 
        WHERE advice_id IN ({placeholders})
        ''', advice_ids)
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

def get_all_advice():
    """Get all valid advice with metadata"""
    with get_db_connection() as conn:
        return pd.read_sql('''
        SELECT 
            a.advice_id,
            a.post_id,
            a.advice_text,
            a.group_name,
            a.created_at,
            p.title,
            p.ups,
            p.upvote_ratio,
            p.subreddit_name_prefixed
        FROM advice a
        JOIN posts p ON a.post_id = p.id
        WHERE a.is_valid = TRUE
        ORDER BY p.ups DESC, a.created_at DESC
        ''', conn)

def export_advice_to_csv(filename):
    """Export advice in a clean format optimized for website display"""
    with get_db_connection() as conn:
        df = pd.read_sql('''
        SELECT 
            a.advice_text as advice,
            a.group_name as category,
            p.ups as upvotes,
            p.upvote_ratio as upvote_ratio,
            p.subreddit_name_prefixed as subreddit
        FROM advice a
        JOIN posts p ON a.post_id = p.id
        WHERE a.is_valid = TRUE
        ORDER BY a.group_name, p.ups DESC
        ''', conn)
    
    if len(df) == 0:
        print("No advice to export")
        return
    
    # Clean for website display
    df['advice'] = df['advice'].str.strip()
    df['category'] = df['category'].str.title()
    
    # Export
    df.to_csv(f"data/{filename}", index=False, encoding='utf-8')