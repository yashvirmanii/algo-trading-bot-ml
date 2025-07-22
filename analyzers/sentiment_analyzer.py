"""
Sentiment Analysis as Supportive Enhancement Layer

This module implements comprehensive sentiment analysis that enhances technical signals
without replacing them. It enables bidirectional trading based on sentiment + technical
alignment and provides weighted integration into the final trading decision.

Core Philosophy:
- Supportive, not replacement: Sentiment enhances technical signals
- Bidirectional trading: Enable both long and short positions
- Weight-based integration: Sentiment gets 15-20% weight in final score
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
import requests
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# NLP libraries
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    from textblob import TextBlob
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available, using basic sentiment analysis")

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class SentimentConfig:
    """Configuration for sentiment analysis"""
    # Sentiment sources
    news_sources: List[str] = field(default_factory=lambda: [
        'economic_times', 'moneycontrol', 'business_standard', 'livemint'
    ])
    social_sources: List[str] = field(default_factory=lambda: [
        'twitter', 'reddit', 'stocktwits'
    ])
    
    # Model configuration
    model_name: str = 'nlptown/bert-base-multilingual-uncased-sentiment'
    use_finbert: bool = True
    fallback_to_textblob: bool = True
    
    # Sentiment weights
    news_weight: float = 0.6  # 60% weight to news
    social_weight: float = 0.4  # 40% weight to social media
    
    # Time windows
    news_lookback_hours: int = 24  # Look back 24 hours for news
    social_lookback_hours: int = 6   # Look back 6 hours for social media
    
    # Sentiment thresholds
    strong_positive_threshold: float = 0.6
    strong_negative_threshold: float = -0.6
    neutral_threshold: float = 0.2
    
    # Integration weights
    sentiment_weight_in_final_score: float = 0.18  # 18% weight in final decision
    technical_weight_in_final_score: float = 0.82  # 82% weight to technical
    
    # Caching
    cache_duration_minutes: int = 30
    max_cache_size: int = 1000


@dataclass
class SentimentResult:
    """Result of sentiment analysis"""
    symbol: str
    overall_sentiment: float  # -1 to +1
    confidence: float  # 0 to 1
    news_sentiment: float
    social_sentiment: float
    sentiment_category: str  # 'strong_positive', 'positive', 'neutral', 'negative', 'strong_negative'
    sources_analyzed: int
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Detailed breakdown
    news_articles: List[Dict] = field(default_factory=list)
    social_posts: List[Dict] = field(default_factory=list)
    sentiment_history: List[float] = field(default_factory=list)
    
    # Trading implications
    trading_signal: str = 'neutral'  # 'bullish', 'bearish', 'neutral'
    position_size_multiplier: float = 1.0  # Multiplier for position sizing
    short_opportunity: bool = False  # Whether this enables short sellingc
lass NewsDataProvider:
    """Provides news data from various sources"""
    
    def __init__(self, config: SentimentConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_news_for_symbol(self, symbol: str) -> List[Dict]:
        """Get news articles for a specific symbol"""
        try:
            news_articles = []
            
            # Economic Times API (example - would need actual API)
            et_articles = self._get_economic_times_news(symbol)
            news_articles.extend(et_articles)
            
            # Moneycontrol news (example)
            mc_articles = self._get_moneycontrol_news(symbol)
            news_articles.extend(mc_articles)
            
            # Filter by time window
            cutoff_time = datetime.now() - timedelta(hours=self.config.news_lookback_hours)
            recent_articles = [
                article for article in news_articles 
                if article.get('timestamp', datetime.now()) > cutoff_time
            ]
            
            return recent_articles[:20]  # Limit to 20 most recent articles
            
        except Exception as e:
            logger.error(f"Error getting news for {symbol}: {e}")
            return []
    
    def _get_economic_times_news(self, symbol: str) -> List[Dict]:
        """Get news from Economic Times (placeholder implementation)"""
        # This would integrate with actual Economic Times API
        # For now, return mock data
        return [
            {
                'title': f'{symbol} shows strong quarterly results',
                'content': f'{symbol} reported better than expected earnings with strong growth prospects.',
                'source': 'Economic Times',
                'timestamp': datetime.now() - timedelta(hours=2),
                'url': f'https://economictimes.com/news/{symbol.lower()}'
            }
        ]
    
    def _get_moneycontrol_news(self, symbol: str) -> List[Dict]:
        """Get news from Moneycontrol (placeholder implementation)"""
        # This would integrate with actual Moneycontrol API
        return [
            {
                'title': f'{symbol} faces regulatory challenges',
                'content': f'New regulations may impact {symbol} business operations in the coming quarter.',
                'source': 'Moneycontrol',
                'timestamp': datetime.now() - timedelta(hours=4),
                'url': f'https://moneycontrol.com/news/{symbol.lower()}'
            }
        ]


class SocialMediaProvider:
    """Provides social media sentiment data"""
    
    def __init__(self, config: SentimentConfig):
        self.config = config
    
    def get_social_sentiment(self, symbol: str) -> List[Dict]:
        """Get social media posts for sentiment analysis"""
        try:
            social_posts = []
            
            # Twitter/X posts (would need Twitter API)
            twitter_posts = self._get_twitter_posts(symbol)
            social_posts.extend(twitter_posts)
            
            # Reddit posts (would need Reddit API)
            reddit_posts = self._get_reddit_posts(symbol)
            social_posts.extend(reddit_posts)
            
            # Filter by time window
            cutoff_time = datetime.now() - timedelta(hours=self.config.social_lookback_hours)
            recent_posts = [
                post for post in social_posts 
                if post.get('timestamp', datetime.now()) > cutoff_time
            ]
            
            return recent_posts[:50]  # Limit to 50 most recent posts
            
        except Exception as e:
            logger.error(f"Error getting social media data for {symbol}: {e}")
            return []
    
    def _get_twitter_posts(self, symbol: str) -> List[Dict]:
        """Get Twitter/X posts (placeholder implementation)"""
        # This would integrate with Twitter API v2
        return [
            {
                'text': f'$${symbol} looking bullish with strong volume breakout! 🚀',
                'source': 'Twitter',
                'timestamp': datetime.now() - timedelta(minutes=30),
                'engagement': 45,
                'user_followers': 1200
            },
            {
                'text': f'Concerned about $${symbol} recent price action, might be overvalued',
                'source': 'Twitter',
                'timestamp': datetime.now() - timedelta(hours=1),
                'engagement': 23,
                'user_followers': 800
            }
        ]
    
    def _get_reddit_posts(self, symbol: str) -> List[Dict]:
        """Get Reddit posts (placeholder implementation)"""
        # This would integrate with Reddit API
        return [
            {
                'text': f'DD on {symbol}: Strong fundamentals but technical looks weak',
                'source': 'Reddit',
                'subreddit': 'IndiaInvestments',
                'timestamp': datetime.now() - timedelta(hours=2),
                'upvotes': 156,
                'comments': 34
            }
        ]


class SentimentAnalysisEngine:
    """Core sentiment analysis engine using NLP models"""
    
    def __init__(self, config: SentimentConfig):
        self.config = config
        self.sentiment_pipeline = None
        self.tokenizer = None
        self.model = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize NLP models for sentiment analysis"""
        try:
            if TRANSFORMERS_AVAILABLE and self.config.use_finbert:
                # Try to load FinBERT or financial sentiment model
                try:
                    self.sentiment_pipeline = pipeline(
                        "sentiment-analysis",
                        model=self.config.model_name,
                        tokenizer=self.config.model_name
                    )
                    logger.info(f"Loaded sentiment model: {self.config.model_name}")
                except Exception as e:
                    logger.warning(f"Could not load {self.config.model_name}, using default: {e}")
                    self.sentiment_pipeline = pipeline("sentiment-analysis")
            
            elif self.config.fallback_to_textblob:
                logger.info("Using TextBlob for sentiment analysis")
            
        except Exception as e:
            logger.error(f"Error initializing sentiment models: {e}")
    
    def analyze_text_sentiment(self, text: str) -> Tuple[float, float]:
        """
        Analyze sentiment of a single text
        
        Returns:
            Tuple of (sentiment_score, confidence)
            sentiment_score: -1 (very negative) to +1 (very positive)
            confidence: 0 to 1
        """
        try:
            if self.sentiment_pipeline:
                # Use transformer model
                result = self.sentiment_pipeline(text[:512])  # Limit text length
                
                if isinstance(result, list):
                    result = result[0]
                
                label = result['label'].upper()
                score = result['score']
                
                # Convert to -1 to +1 scale
                if 'POSITIVE' in label or 'POS' in label:
                    sentiment_score = score
                elif 'NEGATIVE' in label or 'NEG' in label:
                    sentiment_score = -score
                else:
                    sentiment_score = 0.0
                
                return sentiment_score, score
                
            else:
                # Fallback to TextBlob
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity  # -1 to +1
                confidence = abs(polarity)  # Use absolute value as confidence
                
                return polarity, confidence
                
        except Exception as e:
            logger.error(f"Error analyzing text sentiment: {e}")
            return 0.0, 0.0
    
    def analyze_batch_sentiment(self, texts: List[str]) -> List[Tuple[float, float]]:
        """Analyze sentiment for multiple texts efficiently"""
        try:
            results = []
            
            # Process in batches for efficiency
            batch_size = 10
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                for text in batch:
                    sentiment, confidence = self.analyze_text_sentiment(text)
                    results.append((sentiment, confidence))
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch sentiment analysis: {e}")
            return [(0.0, 0.0)] * len(texts)


class SentimentAnalyzer:
    """
    Main sentiment analyzer that integrates news and social media sentiment
    """
    
    def __init__(self, config: SentimentConfig = None):
        self.config = config or SentimentConfig()
        
        # Initialize components
        self.news_provider = NewsDataProvider(self.config)
        self.social_provider = SocialMediaProvider(self.config)
        self.sentiment_engine = SentimentAnalysisEngine(self.config)
        
        # Caching
        self.sentiment_cache = {}
        self.cache_timestamps = {}
        
        # Performance tracking
        self.analysis_history = deque(maxlen=1000)
        
        logger.info("SentimentAnalyzer initialized")
    
    def analyze_symbol_sentiment(self, symbol: str, use_cache: bool = True) -> SentimentResult:
        """
        Analyze comprehensive sentiment for a symbol
        
        Args:
            symbol: Stock symbol to analyze
            use_cache: Whether to use cached results
            
        Returns:
            SentimentResult with comprehensive analysis
        """
        try:
            # Check cache first
            if use_cache and self._is_cache_valid(symbol):
                logger.debug(f"Using cached sentiment for {symbol}")
                return self.sentiment_cache[symbol]
            
            start_time = time.time()
            
            # Get news data
            news_articles = self.news_provider.get_news_for_symbol(symbol)
            
            # Get social media data
            social_posts = self.social_provider.get_social_sentiment(symbol)
            
            # Analyze news sentiment
            news_sentiment, news_confidence = self._analyze_news_sentiment(news_articles)
            
            # Analyze social sentiment
            social_sentiment, social_confidence = self._analyze_social_sentiment(social_posts)
            
            # Combine sentiments
            overall_sentiment = (
                news_sentiment * self.config.news_weight +
                social_sentiment * self.config.social_weight
            )
            
            # Calculate overall confidence
            overall_confidence = (
                news_confidence * self.config.news_weight +
                social_confidence * self.config.social_weight
            )
            
            # Categorize sentiment
            sentiment_category = self._categorize_sentiment(overall_sentiment)
            
            # Determine trading implications
            trading_signal, position_multiplier, short_opportunity = self._determine_trading_implications(
                overall_sentiment, overall_confidence
            )
            
            # Create result
            result = SentimentResult(
                symbol=symbol,
                overall_sentiment=overall_sentiment,
                confidence=overall_confidence,
                news_sentiment=news_sentiment,
                social_sentiment=social_sentiment,
                sentiment_category=sentiment_category,
                sources_analyzed=len(news_articles) + len(social_posts),
                news_articles=news_articles,
                social_posts=social_posts,
                trading_signal=trading_signal,
                position_size_multiplier=position_multiplier,
                short_opportunity=short_opportunity
            )
            
            # Cache result
            self._cache_result(symbol, result)
            
            # Track performance
            analysis_time = time.time() - start_time
            self._track_analysis_performance(symbol, analysis_time, len(news_articles) + len(social_posts))
            
            logger.info(f"Sentiment analysis for {symbol}: {overall_sentiment:.3f} ({sentiment_category}) in {analysis_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment for {symbol}: {e}")
            return self._get_default_sentiment_result(symbol)
    
    def _analyze_news_sentiment(self, news_articles: List[Dict]) -> Tuple[float, float]:
        """Analyze sentiment from news articles"""
        if not news_articles:
            return 0.0, 0.0
        
        try:
            # Extract text from articles
            texts = []
            for article in news_articles:
                title = article.get('title', '')
                content = article.get('content', '')
                combined_text = f"{title}. {content}"
                texts.append(combined_text)
            
            # Analyze sentiment
            sentiment_results = self.sentiment_engine.analyze_batch_sentiment(texts)
            
            # Calculate weighted average (more recent articles get higher weight)
            total_sentiment = 0.0
            total_weight = 0.0
            total_confidence = 0.0
            
            for i, (sentiment, confidence) in enumerate(sentiment_results):
                # Weight decreases with age (more recent = higher weight)
                weight = 1.0 / (i + 1)
                total_sentiment += sentiment * weight * confidence
                total_weight += weight * confidence
                total_confidence += confidence
            
            if total_weight > 0:
                avg_sentiment = total_sentiment / total_weight
                avg_confidence = total_confidence / len(sentiment_results)
                return avg_sentiment, avg_confidence
            else:
                return 0.0, 0.0
                
        except Exception as e:
            logger.error(f"Error analyzing news sentiment: {e}")
            return 0.0, 0.0
    
    def _analyze_social_sentiment(self, social_posts: List[Dict]) -> Tuple[float, float]:
        """Analyze sentiment from social media posts"""
        if not social_posts:
            return 0.0, 0.0
        
        try:
            # Extract text from posts
            texts = [post.get('text', '') for post in social_posts]
            
            # Analyze sentiment
            sentiment_results = self.sentiment_engine.analyze_batch_sentiment(texts)
            
            # Calculate weighted average (engagement-weighted)
            total_sentiment = 0.0
            total_weight = 0.0
            total_confidence = 0.0
            
            for i, (sentiment, confidence) in enumerate(sentiment_results):
                post = social_posts[i]
                
                # Weight by engagement (likes, retweets, upvotes, etc.)
                engagement = post.get('engagement', post.get('upvotes', 1))
                weight = np.log(engagement + 1)  # Log scale to prevent extreme weights
                
                total_sentiment += sentiment * weight * confidence
                total_weight += weight * confidence
                total_confidence += confidence
            
            if total_weight > 0:
                avg_sentiment = total_sentiment / total_weight
                avg_confidence = total_confidence / len(sentiment_results)
                return avg_sentiment, avg_confidence
            else:
                return 0.0, 0.0
                
        except Exception as e:
            logger.error(f"Error analyzing social sentiment: {e}")
            return 0.0, 0.0
    
    def _categorize_sentiment(self, sentiment: float) -> str:
        """Categorize sentiment into discrete categories"""
        if sentiment >= self.config.strong_positive_threshold:
            return 'strong_positive'
        elif sentiment >= self.config.neutral_threshold:
            return 'positive'
        elif sentiment <= self.config.strong_negative_threshold:
            return 'strong_negative'
        elif sentiment <= -self.config.neutral_threshold:
            return 'negative'
        else:
            return 'neutral'
    
    def _determine_trading_implications(self, sentiment: float, confidence: float) -> Tuple[str, float, bool]:
        """Determine trading implications from sentiment"""
        # Trading signal
        if sentiment >= self.config.strong_positive_threshold and confidence > 0.6:
            trading_signal = 'bullish'
            position_multiplier = 1.0 + (sentiment * 0.5)  # Up to 50% increase
            short_opportunity = False
        elif sentiment <= self.config.strong_negative_threshold and confidence > 0.6:
            trading_signal = 'bearish'
            position_multiplier = 1.0 - (abs(sentiment) * 0.3)  # Up to 30% decrease for longs
            short_opportunity = True  # Enable short selling
        else:
            trading_signal = 'neutral'
            position_multiplier = 1.0
            short_opportunity = False
        
        return trading_signal, position_multiplier, short_opportunity
    
    def _is_cache_valid(self, symbol: str) -> bool:
        """Check if cached sentiment is still valid"""
        if symbol not in self.sentiment_cache:
            return False
        
        cache_time = self.cache_timestamps.get(symbol)
        if not cache_time:
            return False
        
        age_minutes = (datetime.now() - cache_time).total_seconds() / 60
        return age_minutes < self.config.cache_duration_minutes
    
    def _cache_result(self, symbol: str, result: SentimentResult):
        """Cache sentiment result"""
        self.sentiment_cache[symbol] = result
        self.cache_timestamps[symbol] = datetime.now()
        
        # Clean old cache entries
        if len(self.sentiment_cache) > self.config.max_cache_size:
            oldest_symbol = min(self.cache_timestamps.keys(), 
                              key=lambda k: self.cache_timestamps[k])
            del self.sentiment_cache[oldest_symbol]
            del self.cache_timestamps[oldest_symbol]
    
    def _track_analysis_performance(self, symbol: str, analysis_time: float, sources_count: int):
        """Track sentiment analysis performance"""
        performance_record = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'analysis_time': analysis_time,
            'sources_analyzed': sources_count
        }
        self.analysis_history.append(performance_record)
    
    def _get_default_sentiment_result(self, symbol: str) -> SentimentResult:
        """Get default sentiment result when analysis fails"""
        return SentimentResult(
            symbol=symbol,
            overall_sentiment=0.0,
            confidence=0.0,
            news_sentiment=0.0,
            social_sentiment=0.0,
            sentiment_category='neutral',
            sources_analyzed=0,
            trading_signal='neutral',
            position_size_multiplier=1.0,
            short_opportunity=False
        )
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get sentiment analysis performance statistics"""
        if not self.analysis_history:
            return {'message': 'No analysis history available'}
        
        recent_analyses = list(self.analysis_history)[-50:]  # Last 50 analyses
        
        return {
            'total_analyses': len(self.analysis_history),
            'average_analysis_time': np.mean([a['analysis_time'] for a in recent_analyses]),
            'average_sources_per_analysis': np.mean([a['sources_analyzed'] for a in recent_analyses]),
            'cache_hit_rate': len(self.sentiment_cache) / len(self.analysis_history) if self.analysis_history else 0,
            'cached_symbols': len(self.sentiment_cache),
            'config': {
                'news_weight': self.config.news_weight,
                'social_weight': self.config.social_weight,
                'sentiment_weight_in_final_score': self.config.sentiment_weight_in_final_score
            }
        }


# Example usage and testing
if __name__ == "__main__":
    # Create sentiment analyzer
    config = SentimentConfig(
        sentiment_weight_in_final_score=0.18,
        strong_positive_threshold=0.6,
        strong_negative_threshold=-0.6
    )
    
    analyzer = SentimentAnalyzer(config)
    
    # Test sentiment analysis
    test_symbols = ['RELIANCE', 'TCS', 'INFY']
    
    for symbol in test_symbols:
        result = analyzer.analyze_symbol_sentiment(symbol)
        print(f"\n{symbol} Sentiment Analysis:")
        print(f"Overall Sentiment: {result.overall_sentiment:.3f} ({result.sentiment_category})")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Trading Signal: {result.trading_signal}")
        print(f"Position Multiplier: {result.position_size_multiplier:.2f}")
        print(f"Short Opportunity: {result.short_opportunity}")
        print(f"Sources Analyzed: {result.sources_analyzed}")
    
    # Get statistics
    stats = analyzer.get_analysis_statistics()
    print(f"\nAnalysis Statistics: {stats}")