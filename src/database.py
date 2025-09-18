import sqlite3
import pandas as pd
from datetime import datetime
from contextlib import contextmanager
import os

DATABASE_PATH = "data/advice_aggregator.db"

def init_database():
    """Initialize the database with required tables"""
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
            scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            processed_for_advice BOOLEAN DEFAULT FALSE
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
            post_id TEXT PRIMARY KEY,
            status TEXT, -- 'scraped', 'processing', 'completed', 'failed'
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            error_message TEXT,
            FOREIGN KEY (post_id) REFERENCES posts(id)
        );

        -- Create indexes for better performance
        CREATE INDEX IF NOT EXISTS idx_posts_processed ON posts(processed_for_advice);
        CREATE INDEX IF NOT EXISTS idx_advice_group ON advice(group_name);
        CREATE INDEX IF NOT EXISTS idx_advice_post ON advice(post_id);
        CREATE INDEX IF NOT EXISTS idx_status ON processing_status(status);
        ''')
        
    print(f"Database initialized at {DATABASE_PATH}")

@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row  # Enable dict-like access
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
        
        print(f"Saved {len(new_posts)} new posts to database")
        return len(new_posts)

def get_unprocessed_posts(limit=None):
    """Get posts that haven't been processed for advice yet"""
    query = '''
    SELECT p.* FROM posts p
    JOIN processing_status ps ON p.id = ps.post_id
    WHERE ps.status IN ('scraped', 'failed')
    AND p.processed_for_advice = FALSE
    ORDER BY p.ups DESC, p.scraped_at ASC
    '''
    
    if limit:
        query += f" LIMIT {limit}"
    
    with get_db_connection() as conn:
        return pd.read_sql(query, conn)

def mark_posts_processing(post_ids):
    """Mark posts as currently being processed"""
    if not post_ids:
        return
        
    with get_db_connection() as conn:
        placeholders = ','.join(['?' for _ in post_ids])
        conn.execute(f'''
        UPDATE processing_status 
        SET status = 'processing', last_updated = CURRENT_TIMESTAMP
        WHERE post_id IN ({placeholders})
        ''', post_ids)
        conn.commit()

def mark_posts_completed(post_ids):
    """Mark posts as successfully processed"""
    if not post_ids:
        return
        
    with get_db_connection() as conn:
        placeholders = ','.join(['?' for _ in post_ids])
        conn.execute(f'''
        UPDATE posts 
        SET processed_for_advice = TRUE
        WHERE id IN ({placeholders})
        ''', post_ids)
        
        conn.execute(f'''
        UPDATE processing_status 
        SET status = 'completed', last_updated = CURRENT_TIMESTAMP
        WHERE post_id IN ({placeholders})
        ''', post_ids)
        conn.commit()

def mark_posts_failed(post_ids, error_message="Processing failed"):
    """Mark posts as failed to process"""
    if not post_ids:
        return
        
    with get_db_connection() as conn:
        for post_id in post_ids:
            conn.execute('''
            UPDATE processing_status 
            SET status = 'failed', error_message = ?, last_updated = CURRENT_TIMESTAMP
            WHERE post_id = ?
            ''', (error_message, post_id))
        conn.commit()

def save_advice_to_db(advice_df):
    """Save extracted advice to database"""
    if len(advice_df) == 0:
        return 0
        
    # Prepare advice data for insertion
    advice_data = advice_df[['id', 'advice', 'group']].copy()
    advice_data.columns = ['post_id', 'advice_text', 'group_name']
    
    with get_db_connection() as conn:
        advice_data.to_sql('advice', conn, if_exists='append', index=False)
        return len(advice_data)

def get_processing_stats():
    """Get current processing statistics"""
    with get_db_connection() as conn:
        stats = pd.read_sql('''
        SELECT 
            ps.status,
            COUNT(*) as count
        FROM processing_status ps
        GROUP BY ps.status
        ''', conn)
        
        advice_stats = pd.read_sql('''
        SELECT 
            COUNT(*) as total_advice,
            COUNT(DISTINCT group_name) as unique_groups,
            COUNT(DISTINCT post_id) as posts_with_advice
        FROM advice
        WHERE is_valid = TRUE
        ''', conn)
        
        return {
            'processing_status': stats.set_index('status')['count'].to_dict(),
            'advice_stats': advice_stats.iloc[0].to_dict()
        }

def get_advice_by_group():
    """Get advice grouped by theme"""
    with get_db_connection() as conn:
        return pd.read_sql('''
        SELECT 
            a.group_name,
            COUNT(*) as advice_count,
            COUNT(DISTINCT a.post_id) as post_count,
            AVG(p.ups) as avg_upvotes
        FROM advice a
        JOIN posts p ON a.post_id = p.id
        WHERE a.is_valid = TRUE
        GROUP BY a.group_name
        ORDER BY advice_count DESC
        ''', conn)

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

if __name__ == "__main__":
    # Initialize database and show stats
    init_database()
    stats = get_processing_stats()
    print("Processing Stats:", stats)