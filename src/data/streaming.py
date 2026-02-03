"""
Real-time Streaming Data Module for Stock AI Predictor
WebSocket-based streaming from Polygon.io, Alpaca, and other providers.
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import AsyncIterator, Callable, Dict, List, Optional, Any
from collections import deque
from enum import Enum
import threading

logger = logging.getLogger(__name__)


class StreamEventType(Enum):
    TRADE = "trade"
    QUOTE = "quote"
    BAR = "bar"
    STATUS = "status"
    ERROR = "error"


@dataclass
class StreamEvent:
    """Base event from streaming data"""
    event_type: StreamEventType
    symbol: str
    timestamp: datetime
    data: Dict[str, Any]


@dataclass
class TradeEvent(StreamEvent):
    """Real-time trade event"""
    price: float
    size: int
    exchange: str = ""
    conditions: List[str] = field(default_factory=list)


@dataclass
class QuoteEvent(StreamEvent):
    """Real-time quote (bid/ask) event"""
    bid_price: float
    bid_size: int
    ask_price: float
    ask_size: int


@dataclass
class BarEvent(StreamEvent):
    """Aggregated bar/candle event"""
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float = 0.0


class StreamBuffer:
    """Thread-safe buffer for streaming data with configurable capacity"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._buffer: deque = deque(maxlen=max_size)
        self._lock = threading.Lock()
        self._subscribers: List[Callable] = []
    
    def add(self, event: StreamEvent):
        with self._lock:
            self._buffer.append(event)
        # Notify subscribers
        for callback in self._subscribers:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Subscriber callback error: {e}")
    
    def get_recent(self, count: int = 100) -> List[StreamEvent]:
        with self._lock:
            return list(self._buffer)[-count:]
    
    def get_by_symbol(self, symbol: str, count: int = 100) -> List[StreamEvent]:
        with self._lock:
            return [e for e in self._buffer if e.symbol == symbol][-count:]
    
    def subscribe(self, callback: Callable[[StreamEvent], None]):
        self._subscribers.append(callback)
    
    def unsubscribe(self, callback: Callable):
        if callback in self._subscribers:
            self._subscribers.remove(callback)
    
    def clear(self):
        with self._lock:
            self._buffer.clear()


class StreamProvider(ABC):
    """Abstract base class for streaming data providers"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.is_connected = False
        self._buffer = StreamBuffer()
        self._subscribed_symbols: set = set()
    
    @abstractmethod
    async def connect(self):
        """Establish WebSocket connection"""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Close connection"""
        pass
    
    @abstractmethod
    async def subscribe(self, symbols: List[str], event_types: List[StreamEventType]):
        """Subscribe to symbols"""
        pass
    
    @abstractmethod
    async def unsubscribe(self, symbols: List[str]):
        """Unsubscribe from symbols"""
        pass
    
    @property
    def buffer(self) -> StreamBuffer:
        return self._buffer


class PolygonStreamProvider(StreamProvider):
    """Polygon.io WebSocket streaming provider"""
    
    CLUSTERS = {
        "stocks": "wss://socket.polygon.io/stocks",
        "options": "wss://socket.polygon.io/options",
        "forex": "wss://socket.polygon.io/forex",
        "crypto": "wss://socket.polygon.io/crypto",
    }
    
    def __init__(self, api_key: str, cluster: str = "stocks"):
        super().__init__(api_key)
        self.cluster = cluster
        self._ws = None
        self._receive_task = None
    
    async def connect(self):
        try:
            import websockets
        except ImportError:
            raise ImportError("websockets library required: pip install websockets")
        
        url = self.CLUSTERS.get(self.cluster, self.CLUSTERS["stocks"])
        logger.info(f"Connecting to Polygon.io {self.cluster} stream...")
        
        self._ws = await websockets.connect(url)
        
        # Authenticate
        auth_msg = {"action": "auth", "params": self.api_key}
        await self._ws.send(json.dumps(auth_msg))
        
        response = await self._ws.recv()
        data = json.loads(response)
        
        if isinstance(data, list) and data[0].get("status") == "auth_success":
            self.is_connected = True
            logger.info("Polygon.io authentication successful")
            
            # Start receive loop
            self._receive_task = asyncio.create_task(self._receive_loop())
        else:
            raise ConnectionError(f"Authentication failed: {data}")
    
    async def disconnect(self):
        if self._receive_task:
            self._receive_task.cancel()
        if self._ws:
            await self._ws.close()
        self.is_connected = False
        logger.info("Disconnected from Polygon.io")
    
    async def subscribe(self, symbols: List[str], event_types: List[StreamEventType]):
        if not self.is_connected:
            raise RuntimeError("Not connected")
        
        params = []
        for symbol in symbols:
            for et in event_types:
                if et == StreamEventType.TRADE:
                    params.append(f"T.{symbol}")
                elif et == StreamEventType.QUOTE:
                    params.append(f"Q.{symbol}")
                elif et == StreamEventType.BAR:
                    params.append(f"A.{symbol}")  # Aggregates
        
        if params:
            msg = {"action": "subscribe", "params": ",".join(params)}
            await self._ws.send(json.dumps(msg))
            self._subscribed_symbols.update(symbols)
            logger.info(f"Subscribed to: {params}")
    
    async def unsubscribe(self, symbols: List[str]):
        if not self.is_connected:
            return
        
        params = []
        for symbol in symbols:
            params.extend([f"T.{symbol}", f"Q.{symbol}", f"A.{symbol}"])
        
        msg = {"action": "unsubscribe", "params": ",".join(params)}
        await self._ws.send(json.dumps(msg))
        self._subscribed_symbols.difference_update(symbols)
    
    async def _receive_loop(self):
        """Main receive loop for WebSocket messages"""
        try:
            async for message in self._ws:
                try:
                    events = json.loads(message)
                    if isinstance(events, list):
                        for event_data in events:
                            event = self._parse_event(event_data)
                            if event:
                                self._buffer.add(event)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON: {message}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Receive loop error: {e}")
            self.is_connected = False
    
    def _parse_event(self, data: Dict) -> Optional[StreamEvent]:
        """Parse Polygon.io event format"""
        ev = data.get("ev")
        
        if ev == "T":  # Trade
            return TradeEvent(
                event_type=StreamEventType.TRADE,
                symbol=data.get("sym", ""),
                timestamp=datetime.fromtimestamp(data.get("t", 0) / 1000),
                data=data,
                price=float(data.get("p", 0)),
                size=int(data.get("s", 0)),
                exchange=data.get("x", ""),
                conditions=data.get("c", []),
            )
        
        elif ev == "Q":  # Quote
            return QuoteEvent(
                event_type=StreamEventType.QUOTE,
                symbol=data.get("sym", ""),
                timestamp=datetime.fromtimestamp(data.get("t", 0) / 1000),
                data=data,
                bid_price=float(data.get("bp", 0)),
                bid_size=int(data.get("bs", 0)),
                ask_price=float(data.get("ap", 0)),
                ask_size=int(data.get("as", 0)),
            )
        
        elif ev == "A":  # Aggregate/Bar
            return BarEvent(
                event_type=StreamEventType.BAR,
                symbol=data.get("sym", ""),
                timestamp=datetime.fromtimestamp(data.get("s", 0) / 1000),
                data=data,
                open=float(data.get("o", 0)),
                high=float(data.get("h", 0)),
                low=float(data.get("l", 0)),
                close=float(data.get("c", 0)),
                volume=int(data.get("v", 0)),
                vwap=float(data.get("vw", 0)),
            )
        
        return None


class AlpacaStreamProvider(StreamProvider):
    """Alpaca Markets WebSocket streaming provider"""
    
    ENDPOINTS = {
        "iex": "wss://stream.data.alpaca.markets/v2/iex",
        "sip": "wss://stream.data.alpaca.markets/v2/sip",
    }
    
    def __init__(self, api_key: str, secret_key: str, feed: str = "iex"):
        super().__init__(api_key)
        self.secret_key = secret_key
        self.feed = feed
        self._ws = None
        self._receive_task = None
    
    async def connect(self):
        try:
            import websockets
        except ImportError:
            raise ImportError("websockets library required")
        
        url = self.ENDPOINTS.get(self.feed, self.ENDPOINTS["iex"])
        logger.info(f"Connecting to Alpaca {self.feed} stream...")
        
        self._ws = await websockets.connect(url)
        
        # Wait for connection message
        response = await self._ws.recv()
        
        # Authenticate
        auth_msg = {
            "action": "auth",
            "key": self.api_key,
            "secret": self.secret_key
        }
        await self._ws.send(json.dumps(auth_msg))
        
        response = await self._ws.recv()
        data = json.loads(response)
        
        if isinstance(data, list) and any(m.get("msg") == "authenticated" for m in data):
            self.is_connected = True
            logger.info("Alpaca authentication successful")
            self._receive_task = asyncio.create_task(self._receive_loop())
        else:
            raise ConnectionError(f"Authentication failed: {data}")
    
    async def disconnect(self):
        if self._receive_task:
            self._receive_task.cancel()
        if self._ws:
            await self._ws.close()
        self.is_connected = False
    
    async def subscribe(self, symbols: List[str], event_types: List[StreamEventType]):
        if not self.is_connected:
            raise RuntimeError("Not connected")
        
        msg = {"action": "subscribe"}
        
        for et in event_types:
            if et == StreamEventType.TRADE:
                msg["trades"] = symbols
            elif et == StreamEventType.QUOTE:
                msg["quotes"] = symbols
            elif et == StreamEventType.BAR:
                msg["bars"] = symbols
        
        await self._ws.send(json.dumps(msg))
        self._subscribed_symbols.update(symbols)
    
    async def unsubscribe(self, symbols: List[str]):
        if not self.is_connected:
            return
        
        msg = {
            "action": "unsubscribe",
            "trades": symbols,
            "quotes": symbols,
            "bars": symbols,
        }
        await self._ws.send(json.dumps(msg))
        self._subscribed_symbols.difference_update(symbols)
    
    async def _receive_loop(self):
        """Receive loop for Alpaca messages"""
        try:
            async for message in self._ws:
                try:
                    events = json.loads(message)
                    if isinstance(events, list):
                        for event_data in events:
                            event = self._parse_event(event_data)
                            if event:
                                self._buffer.add(event)
                except json.JSONDecodeError:
                    pass
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Alpaca receive error: {e}")
            self.is_connected = False
    
    def _parse_event(self, data: Dict) -> Optional[StreamEvent]:
        """Parse Alpaca event format"""
        msg_type = data.get("T")
        
        if msg_type == "t":  # Trade
            return TradeEvent(
                event_type=StreamEventType.TRADE,
                symbol=data.get("S", ""),
                timestamp=datetime.fromisoformat(data.get("t", "").replace("Z", "+00:00")),
                data=data,
                price=float(data.get("p", 0)),
                size=int(data.get("s", 0)),
                exchange=data.get("x", ""),
                conditions=data.get("c", []),
            )
        
        elif msg_type == "q":  # Quote
            return QuoteEvent(
                event_type=StreamEventType.QUOTE,
                symbol=data.get("S", ""),
                timestamp=datetime.fromisoformat(data.get("t", "").replace("Z", "+00:00")),
                data=data,
                bid_price=float(data.get("bp", 0)),
                bid_size=int(data.get("bs", 0)),
                ask_price=float(data.get("ap", 0)),
                ask_size=int(data.get("as", 0)),
            )
        
        elif msg_type == "b":  # Bar
            return BarEvent(
                event_type=StreamEventType.BAR,
                symbol=data.get("S", ""),
                timestamp=datetime.fromisoformat(data.get("t", "").replace("Z", "+00:00")),
                data=data,
                open=float(data.get("o", 0)),
                high=float(data.get("h", 0)),
                low=float(data.get("l", 0)),
                close=float(data.get("c", 0)),
                volume=int(data.get("v", 0)),
                vwap=float(data.get("vw", 0)),
            )
        
        return None


class StreamManager:
    """
    Manages multiple streaming providers and aggregates data.
    Provides a unified interface for real-time market data.
    """
    
    def __init__(self):
        self._providers: Dict[str, StreamProvider] = {}
        self._unified_buffer = StreamBuffer()
        self._running = False
    
    def add_provider(self, name: str, provider: StreamProvider):
        """Add a streaming provider"""
        self._providers[name] = provider
        # Forward events to unified buffer
        provider.buffer.subscribe(lambda e: self._unified_buffer.add(e))
    
    async def connect_all(self):
        """Connect all providers"""
        tasks = [p.connect() for p in self._providers.values()]
        await asyncio.gather(*tasks, return_exceptions=True)
        self._running = True
    
    async def disconnect_all(self):
        """Disconnect all providers"""
        tasks = [p.disconnect() for p in self._providers.values()]
        await asyncio.gather(*tasks, return_exceptions=True)
        self._running = False
    
    async def subscribe(self, symbols: List[str], event_types: List[StreamEventType] = None):
        """Subscribe to symbols across all providers"""
        if event_types is None:
            event_types = [StreamEventType.TRADE, StreamEventType.BAR]
        
        tasks = [
            p.subscribe(symbols, event_types)
            for p in self._providers.values()
            if p.is_connected
        ]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_latest(self, symbol: str, count: int = 1) -> List[StreamEvent]:
        """Get latest events for a symbol"""
        return self._unified_buffer.get_by_symbol(symbol, count)
    
    def subscribe_callback(self, callback: Callable[[StreamEvent], None]):
        """Subscribe to all events with a callback"""
        self._unified_buffer.subscribe(callback)
    
    @property
    def buffer(self) -> StreamBuffer:
        return self._unified_buffer
    
    @property
    def is_running(self) -> bool:
        return self._running


# Factory functions
def create_polygon_stream(api_key: str) -> PolygonStreamProvider:
    return PolygonStreamProvider(api_key)


def create_alpaca_stream(api_key: str, secret_key: str) -> AlpacaStreamProvider:
    return AlpacaStreamProvider(api_key, secret_key)
