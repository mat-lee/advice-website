from analyze_post import *
from webscrape import *

if __name__ == "__main__":
    client = get_openrouter_client()
    sentence_transformer = get_sentence_transformer()

    # Analysis Pipeline:
    # 1) Scrape posts
    # 2) Extract advice

    # 3) Regroup advice category names
    # 4) Drop duplicates in each category
    # 5) Give every piece of advice a "usefulness" score

    # 6) Export csv with every advice over a given threshold

    # ------------------------------

    # Scrape subreddits
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

    # print("Step 1: Scraping subreddits...")

    # total_new = scrape_multiple_subreddits(target_subreddits, time_filter="all", limit=500)

    # # Process unprocessed posts
    # print("Step 2: Processing unprocessed posts...")
    # processed_count = process_unprocessed_posts(client)

    # # Process outdated posts
    # print("Step 2: Processing outdated posts...")
    # process_outdated_advice_posts(client)
    
    # # Regroup advice clusters
    # print("Step 3: Regrouping advice clusters...")
    # regroup_clusters(client, target_groups=None)
    
    # # Filter invalid advice and set invalid
    # print("Step 4: Filtering duplicate advice...")
    # filter_duplicates(sentence_transformer, similarity_threshold=0.95)

    # Score advice
    # print("Step 5: Give advice usefulness score")
    # score_advice_quality(client, rescore_scored_advice=True)

    # # # Export
    print("Step 6: Exporting advice")
    export_advice_to_csv("advice_export.csv", quality_threshold=0.0)
    print("Exported advice!")