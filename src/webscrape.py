import os
import re

import numpy as np
import pandas as pd
import praw

def save_data(df, name):
    df.to_csv(f"{name}.csv", index=False)

def load_data(name):
    df = pd.read_csv(f"{name}.csv")
    return df

def webscrape_reddit():
    data = {}

    def add_to_data(data, s):
        save_stats = ['id', 'selftext', 'subreddit_name_prefixed', 'title', 'ups', 'upvote_ratios']

        for stat in save_stats:
            if stat not in data:
                data[stat] = []
            data[stat].append(getattr(s, stat, None))
        
        return data

    # Initialize the Reddit client
    reddit = praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        username=os.getenv("REDDIT_USERNAME"),
        password=os.getenv("REDDIT_PASSWORD"),
        user_agent=os.getenv("REDDIT_USER_AGENT")
    )

    for s in reddit.subreddit("getdisciplined").top(time_filter="all", limit=100):
        add_to_data(data, s)
    
    data = pd.DataFrame(data)

    return data    

if __name__ == "__main__":
    raw_data = webscrape_reddit()
    save_data(raw_data, "raw_data")