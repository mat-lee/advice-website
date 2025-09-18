#!/usr/bin/env python3
"""
Main orchestrator for the Reddit Advice Aggregator
Handles the complete workflow from scraping to analysis
"""

import sys
import argparse
from webscrape_sql import *
from analyze_post_sql import *
from database import init_database, get_processing_stats, cleanup_failed_processing

def setup_database():
    """Initialize database and show current status"""
    print("Setting up database...")
    init_database()
    
    stats = get_processing_stats()
    if stats['processing_status']:
        print("Database already contains data:")
        show_scraping_stats()
    else:
        print("Database is empty and ready for scraping")

def scrape_default_subreddits():
    """Scrape from a curated list of advice-focused subreddits"""
    target_subreddits = [
        "getdisciplined",      # Primary target
        "productivity",        # Work/productivity advice
        "selfimprovement",     # Personal development
        "getmotivated",        # Motivational advice
        "decidingtobebetter",  # Self-improvement focused
        "lifehacks",          # Practical tips
        "LifeProTips",        # Life advice
        "studytips",          # Academic advice
        "fitness",            # Health/fitness advice
        "personalfinance"     # Financial advice
    ]
    
    print("Scraping from curated advice subreddits...")
    total_new = scrape_multiple_subreddits(
        target_subreddits, 
        time_filter="year",  # Past year for relevancy
        limit=150            # Reasonable limit per subreddit
    )
    
    return total_new

def run_complete_workflow():
    """Run the complete end-to-end workflow"""
    print("=== REDDIT ADVICE AGGREGATOR - COMPLETE WORKFLOW ===\n")
    
    # Step 1: Setup
    setup_database()
    
    # Step 2: Scrape (if needed)
    stats = get_processing_stats()
    total_posts = sum(stats['processing_status'].values()) if stats['processing_status'] else 0
    
    if total_posts < 100:  # If we have very few posts, scrape more
        print(f"\nOnly {total_posts} posts in database, scraping more...")
        new_posts = scrape_default_subreddits()
        if new_posts == 0:
            print("No new posts scraped. Check your Reddit API credentials.")
            return
    else:
        print(f"Database has {total_posts} posts, skipping scraping")
    
    # Step 3: Run analysis pipeline
    print("\nStarting analysis pipeline...")
    run_full_analysis_pipeline()
    
    # Step 4: Export results
    print("\nExporting final results...")
    export_advice_to_csv("reddit_advice_final.csv")
    
    print("\n=== WORKFLOW COMPLETE ===")
    print("Check data/reddit_advice_final.csv for your aggregated advice!")

if __name__ == "__main__":
    # run_full_analysis_pipeline()
    export_advice_to_csv("reddit_advice_final.csv")

