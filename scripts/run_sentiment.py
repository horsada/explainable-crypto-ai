import os
from src.agents.sentiment_scout import RedditSentimentAgent

agent = RedditSentimentAgent(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent="crypto-sentiment-agent"
)

if __name__ == "__main__":
    print(agent.get_summary())
