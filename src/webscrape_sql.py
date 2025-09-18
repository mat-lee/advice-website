import os
from dotenv import load_dotenv
import pandas as pd
import praw
from database import init_database, save_posts_to_db, get_processing_stats

load_dotenv()

def webscrape_reddit(subreddit_name, time_filter="all", limit=250):
    """
    Scrape Reddit posts and save to database, avoiding duplicates
    Returns number of new posts scraped
    """
    print(f"Starting to scrape r/{subreddit_name} (limit: {limit})...")
    
    # Initialize database if needed
    init_database()
    
    # Initialize the Reddit client
    reddit = praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        username=os.getenv("REDDIT_USERNAME"),
        password=os.getenv("REDDIT_PASSWORD"),
        user_agent=os.getenv("REDDIT_USER_AGENT")
    )

    # Scrape posts
    posts_data = []
    save_stats = ['id', 'selftext', 'subreddit_name_prefixed', 'title', 'ups', 'upvote_ratio']

    try:
        for submission in reddit.subreddit(subreddit_name).top(time_filter=time_filter, limit=limit):
            post_data = {}
            for stat in save_stats:
                post_data[stat] = getattr(submission, stat, None)
            posts_data.append(post_data)
            
        print(f"Scraped {len(posts_data)} posts from Reddit")
        
    except Exception as e:
        print(f"Error scraping Reddit: {e}")
        return 0

    if not posts_data:
        print("No posts scraped")
        return 0

    # Convert to DataFrame
    posts_df = pd.DataFrame(posts_data)
    
    # Clean data
    posts_df['selftext'] = posts_df['selftext'].fillna('')
    posts_df['title'] = posts_df['title'].fillna('')
    
    # Filter out deleted/removed posts and posts with no content
    posts_df = posts_df[
        (posts_df['selftext'] != '[deleted]') & 
        (posts_df['selftext'] != '[removed]') &
        (posts_df['title'] != '[deleted]') & 
        (posts_df['title'] != '[removed]') &
        ((posts_df['selftext'].str.len() > 10) | (posts_df['title'].str.len() > 10))
    ].copy()
    
    print(f"After filtering: {len(posts_df)} valid posts")
    
    # Save to database
    new_posts_count = save_posts_to_db(posts_df)
    
    return new_posts_count

def scrape_multiple_subreddits(subreddits, time_filter="all", limit=250):
    """
    Scrape multiple subreddits and save all to database
    """
    total_new_posts = 0
    
    for subreddit in subreddits:
        print(f"\n--- Scraping r/{subreddit} ---")
        try:
            new_posts = webscrape_reddit(subreddit, time_filter, limit)
            total_new_posts += new_posts
            print(f"Added {new_posts} new posts from r/{subreddit}")
        except Exception as e:
            print(f"Failed to scrape r/{subreddit}: {e}")
            continue
    
    print(f"\nTotal new posts scraped: {total_new_posts}")
    return total_new_posts

def show_scraping_stats():
    """Display current scraping and processing statistics"""
    stats = get_processing_stats()
    
    print("\n=== CURRENT DATABASE STATUS ===")
    print("Processing Status:")
    for status, count in stats['processing_status'].items():
        print(f"  {status}: {count}")
    
    print("\nAdvice Stats:")
    for key, value in stats['advice_stats'].items():
        print(f"  {key}: {value}")