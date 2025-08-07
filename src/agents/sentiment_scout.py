import praw
from transformers import pipeline
from typing import List


class RedditSentimentAgent:
    def __init__(self, client_id: str, client_secret: str, user_agent: str):
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        self.classifier = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment"
        )

    def fetch_titles(self, subreddit: str = "CryptoCurrency", limit: int = 10) -> List[str]:
        posts = self.reddit.subreddit(subreddit).hot(limit=limit)
        return [post.title for post in posts]

    def score_sentiment(self, texts: List[str]) -> List[float]:
        scores = []
        for text in texts:
            result = self.classifier(text)[0]
            label = result["label"].upper()
            weight = {"NEGATIVE": -1, "NEUTRAL": 0, "POSITIVE": 1}.get(label, 0)
            scores.append(weight * result["score"])
        return scores

    def get_summary(self) -> str:
        titles = self.fetch_titles()
        scores = self.score_sentiment(titles)
        if not scores:
            return "No sentiment data available."

        avg = sum(scores) / len(scores)
        mood = "Bullish" if avg > 0.2 else "Bearish" if avg < -0.2 else "Neutral"
        return f"Avg sentiment score: {round(avg, 3)} — {mood}"
