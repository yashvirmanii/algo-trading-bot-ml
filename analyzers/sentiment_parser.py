"""
Sentiment Parser Module

Fetches news headlines or tweets for each stock and classifies sentiment using a Hugging Face model.
Returns standardized sentiment scores: +1 (Bullish), 0 (Neutral), -1 (Bearish).
"""
import requests
from transformers import pipeline

class SentimentParser:
    def __init__(self, model_name='finiteautomata/twitter-roberta-base-sentiment-analysis', api_key=None):
        self.sentiment_model = pipeline('sentiment-analysis', model=model_name)
        self.api_key = api_key  # For NewsAPI or Twitter API

    def fetch_news(self, symbol):
        # Example: Use NewsAPI to fetch headlines
        url = f'https://newsapi.org/v2/everything?q={symbol}%20stock&apiKey={self.api_key}'
        resp = requests.get(url)
        if resp.status_code == 200:
            articles = resp.json().get('articles', [])
            return [a['title'] for a in articles]
        return []

    def classify_sentiment(self, texts):
        # Run sentiment model on list of texts
        results = self.sentiment_model(texts)
        scores = []
        for r in results:
            label = r['label'].lower()
            if 'positive' in label or 'bullish' in label:
                scores.append(1)
            elif 'negative' in label or 'bearish' in label:
                scores.append(-1)
            else:
                scores.append(0)
        # Majority vote
        if not scores:
            return 0
        avg = sum(scores) / len(scores)
        if avg > 0.2:
            return 1
        elif avg < -0.2:
            return -1
        else:
            return 0

    def get_sentiment(self, symbol):
        headlines = self.fetch_news(symbol)
        return self.classify_sentiment(headlines) 