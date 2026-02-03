"""
Stock Market Sentiment Analysis Module

Analyzes news articles and social media sentiment to enhance
stock price predictions with market mood indicators.

Features:
- Multi-source news aggregation (NewsAPI, Finnhub, Alpha Vantage)
- Social media sentiment from Twitter/X, Reddit, StockTwits
- Transformer-based sentiment classification
- Entity extraction for ticker mentions
- Sentiment time series with decay weighting
- Integration with price prediction models
"""

import asyncio
import logging
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

import aiohttp
import numpy as np

logger = logging.getLogger(__name__)


class SentimentLabel(Enum):
    """Sentiment classification labels"""
    VERY_BEARISH = -2
    BEARISH = -1
    NEUTRAL = 0
    BULLISH = 1
    VERY_BULLISH = 2


@dataclass
class SentimentItem:
    """A single sentiment data point"""
    source: str
    text: str
    sentiment: SentimentLabel
    confidence: float
    tickers: list[str]
    timestamp: datetime
    url: Optional[str] = None
    author: Optional[str] = None
    engagement: int = 0  # likes, retweets, upvotes etc.
    
    @property
    def weighted_score(self) -> float:
        """Get sentiment score weighted by confidence and engagement"""
        base_score = self.sentiment.value * self.confidence
        engagement_boost = min(np.log1p(self.engagement) / 10, 0.5)
        return base_score * (1 + engagement_boost)


@dataclass
class TickerSentiment:
    """Aggregated sentiment for a specific ticker"""
    ticker: str
    overall_score: float
    overall_label: SentimentLabel
    bullish_count: int
    bearish_count: int
    neutral_count: int
    total_items: int
    confidence: float
    sources: dict[str, int]
    updated_at: datetime
    sentiment_items: list[SentimentItem] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "overall_score": self.overall_score,
            "overall_label": self.overall_label.name,
            "bullish_count": self.bullish_count,
            "bearish_count": self.bearish_count,
            "neutral_count": self.neutral_count,
            "total_items": self.total_items,
            "confidence": self.confidence,
            "sources": self.sources,
            "updated_at": self.updated_at.isoformat()
        }


class SentimentAnalyzer:
    """
    Transformer-based sentiment analysis using a financial domain model.
    Falls back to rule-based analysis if model unavailable.
    """
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._initialized = False
        
        # Rule-based fallback patterns
        self.bullish_patterns = [
            r'\b(buy|bullish|long|calls?|moon|rocket|breakout|surge|rally|soar|pump|gain|profit|beat|exceed|upgrade|undervalued)\b',
            r'\b(strong|growth|positive|optimistic|confident|upside|outperform)\b',
            r'🚀|📈|💰|🔥|💎|🙌',
        ]
        self.bearish_patterns = [
            r'\b(sell|bearish|short|puts?|crash|dump|tank|plunge|drop|fall|miss|downgrade|overvalued)\b',
            r'\b(weak|decline|negative|pessimistic|worried|downside|underperform|bankruptcy|lawsuit)\b',
            r'📉|😱|💩|🔻|⚠️',
        ]
        
        # Financial domain stop words
        self.stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been'}
        
    async def initialize(self):
        """Initialize the sentiment model"""
        try:
            # Try loading FinBERT or similar financial sentiment model
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            
            model_name = "ProsusAI/finbert"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self._initialized = True
            logger.info("FinBERT model loaded successfully")
        except ImportError:
            logger.warning("Transformers not available, using rule-based analysis")
        except Exception as e:
            logger.warning(f"Failed to load FinBERT: {e}, using rule-based analysis")
            
    def analyze(self, text: str) -> tuple[SentimentLabel, float]:
        """Analyze sentiment of text"""
        if self._initialized and self.model:
            return self._analyze_with_model(text)
        return self._analyze_rule_based(text)
    
    def _analyze_with_model(self, text: str) -> tuple[SentimentLabel, float]:
        """Use transformer model for sentiment analysis"""
        import torch
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0]
            
        # FinBERT outputs: negative, neutral, positive
        neg_prob = probs[0].item()
        neu_prob = probs[1].item()
        pos_prob = probs[2].item()
        
        # Map to our labels
        score = pos_prob - neg_prob  # Range: -1 to 1
        confidence = max(probs).item()
        
        if score > 0.5:
            label = SentimentLabel.VERY_BULLISH
        elif score > 0.1:
            label = SentimentLabel.BULLISH
        elif score < -0.5:
            label = SentimentLabel.VERY_BEARISH
        elif score < -0.1:
            label = SentimentLabel.BEARISH
        else:
            label = SentimentLabel.NEUTRAL
            
        return label, confidence
    
    def _analyze_rule_based(self, text: str) -> tuple[SentimentLabel, float]:
        """Rule-based sentiment analysis fallback"""
        text_lower = text.lower()
        
        bullish_score = 0
        bearish_score = 0
        
        for pattern in self.bullish_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            bullish_score += len(matches)
            
        for pattern in self.bearish_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            bearish_score += len(matches)
            
        total = bullish_score + bearish_score
        if total == 0:
            return SentimentLabel.NEUTRAL, 0.5
            
        score = (bullish_score - bearish_score) / max(total, 1)
        confidence = min(0.5 + (total * 0.1), 0.9)
        
        if score > 0.5:
            label = SentimentLabel.VERY_BULLISH
        elif score > 0.2:
            label = SentimentLabel.BULLISH
        elif score < -0.5:
            label = SentimentLabel.VERY_BEARISH
        elif score < -0.2:
            label = SentimentLabel.BEARISH
        else:
            label = SentimentLabel.NEUTRAL
            
        return label, confidence
    
    def extract_tickers(self, text: str) -> list[str]:
        """Extract stock tickers from text"""
        # Match $TICKER or common patterns
        patterns = [
            r'\$([A-Z]{1,5})\b',  # $AAPL format
            r'\b([A-Z]{2,5})\s+(?:stock|shares?|calls?|puts?)\b',  # AAPL stock
            r'(?:buy|sell|long|short)\s+([A-Z]{2,5})\b',  # buy AAPL
        ]
        
        tickers = set()
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            tickers.update(m.upper() for m in matches)
            
        # Filter out common false positives
        false_positives = {'THE', 'A', 'I', 'FOR', 'AND', 'OR', 'IT', 'IS', 'CEO', 'IPO', 'ETF', 'BREAKING', 'NEWS', 'UPDATE'}
        tickers = {t for t in tickers if t not in false_positives}
        
        return list(tickers)


class NewsSource(ABC):
    """Abstract base class for news sources"""
    
    @abstractmethod
    async def fetch_news(self, query: str, from_date: datetime) -> list[dict]:
        pass


class NewsAPISource(NewsSource):
    """NewsAPI.org integration"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2"
        
    async def fetch_news(self, query: str, from_date: datetime) -> list[dict]:
        async with aiohttp.ClientSession() as session:
            params = {
                "q": query,
                "from": from_date.isoformat(),
                "sortBy": "publishedAt",
                "language": "en",
                "apiKey": self.api_key
            }
            
            async with session.get(f"{self.base_url}/everything", params=params) as resp:
                if resp.status != 200:
                    logger.error(f"NewsAPI error: {resp.status}")
                    return []
                    
                data = await resp.json()
                return [
                    {
                        "source": "newsapi",
                        "title": article["title"],
                        "description": article.get("description", ""),
                        "content": article.get("content", ""),
                        "url": article["url"],
                        "published_at": article["publishedAt"],
                        "author": article.get("author")
                    }
                    for article in data.get("articles", [])
                ]


class FinnhubSource(NewsSource):
    """Finnhub.io news integration"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://finnhub.io/api/v1"
        
    async def fetch_news(self, query: str, from_date: datetime) -> list[dict]:
        async with aiohttp.ClientSession() as session:
            # Assume query is a ticker symbol for Finnhub
            params = {
                "symbol": query.upper(),
                "from": from_date.strftime("%Y-%m-%d"),
                "to": datetime.now().strftime("%Y-%m-%d"),
                "token": self.api_key
            }
            
            async with session.get(f"{self.base_url}/company-news", params=params) as resp:
                if resp.status != 200:
                    logger.error(f"Finnhub error: {resp.status}")
                    return []
                    
                data = await resp.json()
                return [
                    {
                        "source": "finnhub",
                        "title": article["headline"],
                        "description": article.get("summary", ""),
                        "content": article.get("summary", ""),
                        "url": article["url"],
                        "published_at": datetime.fromtimestamp(article["datetime"]).isoformat(),
                        "author": article.get("source")
                    }
                    for article in data
                ]


class SocialMediaSource(ABC):
    """Abstract base class for social media sources"""
    
    @abstractmethod
    async def fetch_posts(self, query: str, limit: int) -> list[dict]:
        pass


class RedditSource(SocialMediaSource):
    """Reddit integration for r/wallstreetbets, r/stocks, etc."""
    
    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.base_url = "https://oauth.reddit.com"
        self.subreddits = ["wallstreetbets", "stocks", "investing", "options"]
        self._access_token = None
        
    async def _authenticate(self, session: aiohttp.ClientSession):
        """Get OAuth token"""
        auth = aiohttp.BasicAuth(self.client_id, self.client_secret)
        data = {"grant_type": "client_credentials"}
        
        async with session.post(
            "https://www.reddit.com/api/v1/access_token",
            auth=auth,
            data=data
        ) as resp:
            if resp.status == 200:
                token_data = await resp.json()
                self._access_token = token_data["access_token"]
                
    async def fetch_posts(self, query: str, limit: int = 100) -> list[dict]:
        async with aiohttp.ClientSession() as session:
            await self._authenticate(session)
            
            if not self._access_token:
                return []
                
            results = []
            headers = {"Authorization": f"Bearer {self._access_token}"}
            
            for subreddit in self.subreddits:
                params = {"q": query, "limit": limit // len(self.subreddits), "sort": "new"}
                
                async with session.get(
                    f"{self.base_url}/r/{subreddit}/search",
                    headers=headers,
                    params=params
                ) as resp:
                    if resp.status != 200:
                        continue
                        
                    data = await resp.json()
                    for post in data.get("data", {}).get("children", []):
                        post_data = post["data"]
                        results.append({
                            "source": f"reddit/{subreddit}",
                            "text": f"{post_data['title']} {post_data.get('selftext', '')}",
                            "url": f"https://reddit.com{post_data['permalink']}",
                            "author": post_data["author"],
                            "timestamp": datetime.fromtimestamp(post_data["created_utc"]),
                            "engagement": post_data["score"] + post_data["num_comments"]
                        })
                        
            return results


class StockTwitsSource(SocialMediaSource):
    """StockTwits integration"""
    
    def __init__(self):
        self.base_url = "https://api.stocktwits.com/api/2"
        
    async def fetch_posts(self, query: str, limit: int = 100) -> list[dict]:
        async with aiohttp.ClientSession() as session:
            # StockTwits uses symbol streams
            async with session.get(
                f"{self.base_url}/streams/symbol/{query.upper()}.json"
            ) as resp:
                if resp.status != 200:
                    return []
                    
                data = await resp.json()
                return [
                    {
                        "source": "stocktwits",
                        "text": msg["body"],
                        "url": f"https://stocktwits.com/{msg['user']['username']}/message/{msg['id']}",
                        "author": msg["user"]["username"],
                        "timestamp": datetime.fromisoformat(msg["created_at"].replace("Z", "+00:00")),
                        "engagement": msg.get("likes", {}).get("total", 0),
                        "stocktwits_sentiment": msg.get("entities", {}).get("sentiment", {}).get("basic")
                    }
                    for msg in data.get("messages", [])[:limit]
                ]


class SentimentAggregator:
    """
    Main sentiment aggregation service.
    Combines data from multiple sources and maintains sentiment time series.
    """
    
    def __init__(
        self,
        news_api_key: Optional[str] = None,
        finnhub_key: Optional[str] = None,
        reddit_credentials: Optional[tuple[str, str]] = None
    ):
        self.analyzer = SentimentAnalyzer()
        
        # Initialize sources
        self.news_sources: list[NewsSource] = []
        self.social_sources: list[SocialMediaSource] = []
        
        if news_api_key:
            self.news_sources.append(NewsAPISource(news_api_key))
        if finnhub_key:
            self.news_sources.append(FinnhubSource(finnhub_key))
        if reddit_credentials:
            self.social_sources.append(RedditSource(*reddit_credentials))
        self.social_sources.append(StockTwitsSource())
        
        # Sentiment cache
        self.ticker_sentiment: dict[str, TickerSentiment] = {}
        self.sentiment_history: dict[str, list[tuple[datetime, float]]] = defaultdict(list)
        
        # Decay settings for time-weighted sentiment
        self.decay_half_life_hours = 24  # Sentiment decays by 50% every 24 hours
        
    async def initialize(self):
        """Initialize the sentiment analyzer"""
        await self.analyzer.initialize()
        
    async def update_sentiment(self, tickers: list[str], lookback_days: int = 3):
        """Update sentiment for specified tickers"""
        from_date = datetime.now() - timedelta(days=lookback_days)
        
        for ticker in tickers:
            sentiment_items = []
            
            # Fetch news
            for source in self.news_sources:
                try:
                    articles = await source.fetch_news(ticker, from_date)
                    for article in articles:
                        text = f"{article['title']} {article.get('description', '')}"
                        sentiment, confidence = self.analyzer.analyze(text)
                        extracted_tickers = self.analyzer.extract_tickers(text)
                        
                        if ticker in extracted_tickers or not extracted_tickers:
                            sentiment_items.append(SentimentItem(
                                source=article["source"],
                                text=text[:500],
                                sentiment=sentiment,
                                confidence=confidence,
                                tickers=[ticker] if not extracted_tickers else extracted_tickers,
                                timestamp=datetime.fromisoformat(article["published_at"].replace("Z", "+00:00")),
                                url=article.get("url"),
                                author=article.get("author")
                            ))
                except Exception as e:
                    logger.error(f"Error fetching news from source: {e}")
                    
            # Fetch social media
            for source in self.social_sources:
                try:
                    posts = await source.fetch_posts(ticker)
                    for post in posts:
                        text = post["text"]
                        sentiment, confidence = self.analyzer.analyze(text)
                        
                        sentiment_items.append(SentimentItem(
                            source=post["source"],
                            text=text[:500],
                            sentiment=sentiment,
                            confidence=confidence,
                            tickers=self.analyzer.extract_tickers(text) or [ticker],
                            timestamp=post["timestamp"],
                            url=post.get("url"),
                            author=post.get("author"),
                            engagement=post.get("engagement", 0)
                        ))
                except Exception as e:
                    logger.error(f"Error fetching social media from source: {e}")
                    
            # Aggregate sentiment
            if sentiment_items:
                self.ticker_sentiment[ticker] = self._aggregate_ticker_sentiment(ticker, sentiment_items)
                
                # Update history
                score = self.ticker_sentiment[ticker].overall_score
                self.sentiment_history[ticker].append((datetime.now(), score))
                
                # Trim old history
                cutoff = datetime.now() - timedelta(days=30)
                self.sentiment_history[ticker] = [
                    (ts, s) for ts, s in self.sentiment_history[ticker] if ts > cutoff
                ]
                
    def _aggregate_ticker_sentiment(
        self,
        ticker: str,
        items: list[SentimentItem]
    ) -> TickerSentiment:
        """Aggregate sentiment items into ticker sentiment"""
        now = datetime.now()
        
        # Apply time decay weighting
        weighted_scores = []
        weights = []
        
        for item in items:
            age_hours = (now - item.timestamp).total_seconds() / 3600
            decay = 0.5 ** (age_hours / self.decay_half_life_hours)
            
            weight = decay * item.confidence * (1 + min(item.engagement / 100, 1))
            weighted_scores.append(item.weighted_score * weight)
            weights.append(weight)
            
        overall_score = sum(weighted_scores) / max(sum(weights), 1)
        
        # Count by sentiment
        bullish = sum(1 for i in items if i.sentiment.value > 0)
        bearish = sum(1 for i in items if i.sentiment.value < 0)
        neutral = sum(1 for i in items if i.sentiment.value == 0)
        
        # Count by source
        sources = defaultdict(int)
        for item in items:
            source_name = item.source.split("/")[0]
            sources[source_name] += 1
            
        # Determine overall label
        if overall_score > 0.5:
            label = SentimentLabel.VERY_BULLISH
        elif overall_score > 0.1:
            label = SentimentLabel.BULLISH
        elif overall_score < -0.5:
            label = SentimentLabel.VERY_BEARISH
        elif overall_score < -0.1:
            label = SentimentLabel.BEARISH
        else:
            label = SentimentLabel.NEUTRAL
            
        return TickerSentiment(
            ticker=ticker,
            overall_score=overall_score,
            overall_label=label,
            bullish_count=bullish,
            bearish_count=bearish,
            neutral_count=neutral,
            total_items=len(items),
            confidence=sum(i.confidence for i in items) / len(items),
            sources=dict(sources),
            updated_at=now,
            sentiment_items=items[:100]  # Keep top 100 items
        )
        
    def get_sentiment(self, ticker: str) -> Optional[TickerSentiment]:
        """Get current sentiment for a ticker"""
        return self.ticker_sentiment.get(ticker)
    
    def get_sentiment_timeseries(
        self,
        ticker: str,
        days: int = 7
    ) -> list[tuple[datetime, float]]:
        """Get sentiment time series for a ticker"""
        cutoff = datetime.now() - timedelta(days=days)
        return [
            (ts, score) for ts, score in self.sentiment_history.get(ticker, [])
            if ts > cutoff
        ]
    
    def get_sentiment_features(self, ticker: str) -> dict:
        """
        Get sentiment features for ML model integration.
        Returns features that can be used alongside price data.
        """
        sentiment = self.ticker_sentiment.get(ticker)
        history = self.sentiment_history.get(ticker, [])
        
        if not sentiment:
            return {
                "sentiment_score": 0.0,
                "sentiment_momentum": 0.0,
                "sentiment_volatility": 0.0,
                "bullish_ratio": 0.5,
                "news_volume": 0,
                "social_engagement": 0.0
            }
            
        # Calculate momentum (change in sentiment)
        momentum = 0.0
        if len(history) >= 2:
            recent = [s for ts, s in history[-5:]]
            old = [s for ts, s in history[-10:-5]] if len(history) >= 10 else recent[:1]
            momentum = np.mean(recent) - np.mean(old)
            
        # Calculate volatility
        volatility = np.std([s for _, s in history]) if len(history) > 1 else 0.0
        
        # Calculate engagement
        engagement = np.mean([i.engagement for i in sentiment.sentiment_items]) if sentiment.sentiment_items else 0.0
        
        return {
            "sentiment_score": sentiment.overall_score,
            "sentiment_momentum": momentum,
            "sentiment_volatility": volatility,
            "bullish_ratio": sentiment.bullish_count / max(sentiment.total_items, 1),
            "news_volume": sentiment.total_items,
            "social_engagement": min(engagement / 100, 1.0)  # Normalized
        }
    
    async def get_top_movers(self, tickers: list[str]) -> dict[str, list[str]]:
        """Get tickers with strongest sentiment signals"""
        await self.update_sentiment(tickers)
        
        sorted_by_sentiment = sorted(
            [(t, s) for t, s in self.ticker_sentiment.items() if t in tickers],
            key=lambda x: x[1].overall_score
        )
        
        return {
            "most_bullish": [t for t, _ in sorted_by_sentiment[-5:][::-1]],
            "most_bearish": [t for t, _ in sorted_by_sentiment[:5]],
            "highest_volume": sorted(
                [(t, s) for t, s in self.ticker_sentiment.items() if t in tickers],
                key=lambda x: x[1].total_items,
                reverse=True
            )[:5]
        }
