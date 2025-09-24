from analyze_post import *
from webscrape import *

if __name__ == "__main__":
    # Scrape
    target_subreddits = [
        "getdisciplined",      # Primary target
        "productivity",        # Work/productivity advice
        # "selfimprovement",     # Personal development
        # "getmotivated",        # Motivational advice
        # "decidingtobebetter",  # Self-improvement focused
        # "lifehacks",          # Practical tips
        # "LifeProTips",        # Life advice
        # "studytips",          # Academic advice
        # "fitness",            # Health/fitness advice
        # "personalfinance"     # Financial advice
    ]

    total_new = scrape_multiple_subreddits(target_subreddits, time_filter="all", limit=5)

    # Analyze
    run_full_analysis_pipeline()

    # Export
    export_advice_to_csv("advice_export.csv")
    print("Exported advice!")