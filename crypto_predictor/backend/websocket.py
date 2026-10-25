import logging
from typing import Dict, Any, Optional
from datetime import datetime
import threading
import time
from flask import Flask
from flask_socketio import SocketIO, emit, join_room, leave_room

logger = logging.getLogger(__name__)

# Global SocketIO instance
socketio: Optional[SocketIO] = None


def init_socketio(app: Flask, **kwargs) -> SocketIO:
    """Initialize the global SocketIO instance and register handlers."""
    global socketio

    default_config = {
        'cors_allowed_origins': '*',
        'async_mode': 'threading',  # fine for dev; can switch to eventlet/gevent later
        'ping_timeout': 60,
        'ping_interval': 25,
    }
    default_config.update(kwargs)

    socketio = SocketIO(app, **default_config)

    # Register event handlers
    register_handlers(socketio)

    logger.info("WebSocket support initialized")
    return socketio


def get_socketio() -> Optional[SocketIO]:
    """Get the global SocketIO instance."""
    return socketio


def register_handlers(sio: SocketIO):
    """Register WebSocket event handlers."""

    @sio.on('connect')
    def handle_connect():
        logger.info("Client connected")
        emit('connected', {'status': 'ok', 'timestamp': datetime.now().isoformat()})

    @sio.on('disconnect')
    def handle_disconnect():
        logger.info("Client disconnected")

    @sio.on('subscribe_asset')
    def handle_subscribe(data):
        """Subscribe to price updates for a specific asset."""
        data = data or {}
        symbol = data.get('symbol', '').upper()
        if symbol:
            join_room(f"asset_{symbol}")
            emit('subscribed', {'symbol': symbol, 'status': 'ok'})
            logger.info(f"Client subscribed to {symbol}")

    @sio.on('unsubscribe_asset')
    def handle_unsubscribe(data):
        """Unsubscribe from price updates."""
        data = data or {}
        symbol = data.get('symbol', '').upper()
        if symbol:
            leave_room(f"asset_{symbol}")
            emit('unsubscribed', {'symbol': symbol, 'status': 'ok'})
            logger.info(f"Client unsubscribed from {symbol}")

    @sio.on('subscribe_training')
    def handle_training_subscribe(data):
        """Subscribe to training progress updates."""
        data = data or {}
        session_id = data.get('session_id')
        if session_id:
            join_room(f"training_{session_id}")
            emit('training_subscribed', {'session_id': session_id, 'status': 'ok'})
            logger.info(f"Client subscribed to training session {session_id}")


# Event emitter functions
def emit_price_update(symbol: str, price: float, change_24h: Optional[float] = None):
    if socketio:
        data = {
            'symbol': symbol,
            'price': price,
            'change_24h': change_24h,
            'timestamp': datetime.now().isoformat(),
        }
        socketio.emit('price_update', data, room=f"asset_{symbol}")


def emit_training_progress(
    session_id: str,
    model_type: str,
    epoch: int,
    total_epochs: int,
    loss: float,
    eta_seconds: Optional[int] = None,
):
    if socketio:
        progress = (epoch / total_epochs) * 100 if total_epochs else 0
        data = {
            'session_id': session_id,
            'model_type': model_type,
            'epoch': epoch,
            'total_epochs': total_epochs,
            'progress': round(progress, 1),
            'loss': loss,
            'eta_seconds': eta_seconds,
            'timestamp': datetime.now().isoformat(),
        }
        socketio.emit('training_progress', data, room=f"training_{session_id}")


def emit_training_complete(session_id: str, model_type: str, metrics: Dict[str, Any]):
    if socketio:
        data = {
            'session_id': session_id,
            'model_type': model_type,
            'status': 'complete',
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
        }
        socketio.emit('training_complete', data, room=f"training_{session_id}")


def emit_training_error(session_id: str, model_type: str, error: str):
    if socketio:
        data = {
            'session_id': session_id,
            'model_type': model_type,
            'status': 'error',
            'error': error,
            'timestamp': datetime.now().isoformat(),
        }
        socketio.emit('training_error', data, room=f"training_{session_id}")


class PriceStreamer:
    """Background thread for streaming price updates."""

    def __init__(self, fetch_interval: int = 30):
        self.fetch_interval = fetch_interval
        self.subscribed_symbols: set[str] = set()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Start the price streaming background thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._stream_loop, daemon=True)
        self._thread.start()
        logger.info("Price streamer started")

    def stop(self):
        """Stop the price streaming."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("Price streamer stopped")

    def subscribe(self, symbol: str):
        """Add a symbol to the stream."""
        self.subscribed_symbols.add(symbol.upper())

    def unsubscribe(self, symbol: str):
        """Remove a symbol from the stream."""
        self.subscribed_symbols.discard(symbol.upper())

    def _stream_loop(self):
        """Main streaming loop."""
        while self._running:
            try:
                for symbol in list(self.subscribed_symbols):
                    try:
                        stats = self._get_price_stats(symbol)
                        if stats and stats.get('price') is not None:
                            emit_price_update(
                                symbol,
                                stats['price'],
                                stats.get('change_24h'),
                            )
                    except Exception as e:
                        logger.warning(f"Failed to fetch price for {symbol}: {e}")

                time.sleep(self.fetch_interval)
            except Exception as e:
                logger.error(f"Error in price stream loop: {e}")
                time.sleep(5)

    def _get_price_stats(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get the latest price for a symbol."""
        from .api import get_service

        price = get_service().fetcher.get_latest_price(symbol)
        return {'price': price} if price is not None else None


# Global price streamer instance
_price_streamer: Optional[PriceStreamer] = None


def get_price_streamer() -> PriceStreamer:
    """Get or create the global price streamer."""
    global _price_streamer
    if _price_streamer is None:
        _price_streamer = PriceStreamer()
    return _price_streamer
