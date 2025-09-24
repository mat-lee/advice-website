from analyze_post import *
from webscrape import *

if __name__ == "__main__":
    client = get_openrouter_client()

    # Analysis Pipeline:
    # 1) Scrape posts
    # 2) Extract advice

    # 3) Regroup advice category names
    # 4) Drop duplicates in each category
    # 5) Give every piece of advice a "usefulness" score

    # 6) Export csv with every advice over a given threshold

    # ------------------------------

    # Scrape subreddits
    print("Step 1: Scraping subreddits...")

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

    # total_new = scrape_multiple_subreddits(target_subreddits, time_filter="all", limit=500)

    # Process unprocessed posts
    print("Step 2: Processing unprocessed posts...")
    processed_count = process_unprocessed_posts(client)

    # Process outdated posts
    print("Step 2: Processing outdated posts...")
    process_outdated_advice_posts(client)
    
    # Regroup advice clusters
    print("Step 3: Regrouping advice clusters...")
    regroup_clusters(client, max_groups=None)
    
    # Filter invalid advice
    print("Step 4: Filtering duplicate advice...")
    filter_duplicates(client)

    # Score advice
    print("Step 5: Give advice usefulness score")
    score_advice(client)

    # Export
    print("Step 6: Exporting advice")
    export_advice_to_csv("advice_export.csv")

    print("Exported advice!")