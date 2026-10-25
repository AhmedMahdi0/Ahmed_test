"""
Flask API backend for CryptoPredictor.
Provides REST endpoints for the web frontend with WebSocket support.
"""

import os
import logging
import gzip
import uuid
from datetime import datetime
from functools import wraps
from typing import Optional
from io import BytesIO
from .websocket import init_socketio

from flask import Flask, jsonify, request, send_from_directory, Response
from flask_cors import CORS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our modules
from .config import get_config
from .prediction_service import PredictionService
from .cache import get_cache, init_cache
from .tasks import submit_task, get_task_status, cancel_task

import os
_static_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'frontend')
app = Flask(__name__, static_folder=_static_folder, static_url_path='')
CORS(app)

socketio = None
try:
    socketio = init_socketio(app)
except ImportError:
    logger.warning("WebSocket support not available (install flask-socketio)")

# Global service instance
prediction_service: Optional[PredictionService] = None

# Training sessions for async training
training_sessions = {}

training_status = {
    'is_training': False,
    'current_model': None,
    'progress': 0,
    'message': ''
}

def get_service() -> PredictionService:
    """Get or create the prediction service."""
    global prediction_service
    if prediction_service is None:
        prediction_service = PredictionService()
    return prediction_service


def gzip_response(f):
    """Decorator for gzip compression of responses."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        response = f(*args, **kwargs)
        
        # Check if client accepts gzip
        if 'gzip' not in request.headers.get('Accept-Encoding', ''):
            return response
        
        # Skip if response is already compressed or too small
        if isinstance(response, Response):
            data = response.get_data()
            if len(data) < 500:  # Don't compress small responses
                return response
            
            # Compress the response
            gzip_buffer = BytesIO()
            with gzip.GzipFile(mode='wb', fileobj=gzip_buffer, compresslevel=6) as gz:
                gz.write(data)
            
            compressed = gzip_buffer.getvalue()
            
            # Only use compressed if it's smaller
            if len(compressed) < len(data):
                response.set_data(compressed)
                response.headers['Content-Encoding'] = 'gzip'
                response.headers['Content-Length'] = len(compressed)
        
        return response
    return decorated_function


def handle_errors(f):
    """Decorator for error handling."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ValueError as e:
            logger.error(f"Validation error: {e}")
            return jsonify({'error': str(e), 'type': 'validation'}), 400
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            return jsonify({'error': str(e), 'type': 'server'}), 500
    return decorated_function


# ============================================
# Static file serving (for SPA)
# ============================================

@app.route('/')
def serve_frontend():
    """Serve the main frontend page."""
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    """Serve static files."""
    if os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')


# ============================================
# API Endpoints
# ============================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '2.0.0',
        'features': {
            'websocket': socketio is not None,
            'async_training': True,
            'caching': True
        }
    })


@app.route('/api/cryptocurrencies', methods=['GET'])
@handle_errors
def get_cryptocurrencies():
    """Get list of available cryptocurrencies with market data."""
    service = get_service()
    cryptos = service.get_available_cryptos()
    
    # Add market data if requested
    include_market_data = request.args.get('include_market_data', 'false').lower() == 'true'
    
    if include_market_data:
        for crypto in cryptos:
            try:
                import yfinance as yf
                ticker = f"{crypto['symbol']}-USD"
                data = yf.Ticker(ticker)
                info = data.fast_info
                crypto['price'] = info.get('lastPrice')
                crypto['change_24h'] = info.get('regularMarketChangePercent')
            except Exception:
                pass
    
    return jsonify({
        'cryptocurrencies': cryptos,
        'count': len(cryptos)
    })


@app.route('/api/asset/<symbol>/price', methods=['GET'])
@handle_errors
def get_asset_price(symbol: str):
    """Get current price for an asset."""
    symbol = symbol.upper()
    
    # Try cache first
    cache = get_cache()
    cache_key = f"price:{symbol}"
    cached_price = cache.get(cache_key)
    
    if cached_price:
        return jsonify(cached_price)
    
    # Fetch fresh data
    try:
        import yfinance as yf
        ticker = f"{symbol}-USD"
        data = yf.Ticker(ticker)
        info = data.info
        
        result = {
            'symbol': symbol,
            'price': info.get('regularMarketPrice'),
            'change_24h': info.get('regularMarketChangePercent'),
            'volume_24h': info.get('volume24Hr'),
            'market_cap': info.get('marketCap'),
            'high_24h': info.get('dayHigh'),
            'low_24h': info.get('dayLow'),
            'timestamp': datetime.now().isoformat()
        }
        
        # Cache for 30 seconds
        cache.set(cache_key, result, ttl=30)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'Failed to fetch price: {str(e)}'}), 503


@app.route('/api/asset/<symbol>/sparkline', methods=['GET'])
@handle_errors
def get_asset_sparkline(symbol: str):
    """Get sparkline data for an asset (7-day price history)."""
    symbol = symbol.upper()
    days = int(request.args.get('days', 7))
    
    # Try cache first
    cache = get_cache()
    cache_key = f"sparkline:{symbol}:{days}"
    cached_data = cache.get(cache_key)
    
    if cached_data:
        return jsonify(cached_data)
    
    try:
        import yfinance as yf
        from datetime import timedelta
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        ticker = f"{symbol}-USD"
        data = yf.download(
            ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            progress=False
        )
        
        if data.empty:
            return jsonify({'symbol': symbol, 'prices': [], 'error': 'No data available'})
        
        # Get closing prices and calculate stats
        prices = data['Close'].dropna().tolist()
        
        result = {
            'symbol': symbol,
            'prices': prices,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'change_percent': ((prices[-1] - prices[0]) / prices[0] * 100) if len(prices) > 1 else 0,
            'min_price': min(prices) if prices else 0,
            'max_price': max(prices) if prices else 0
        }
        
        # Cache for 60 minutes
        cache.set(cache_key, result, ttl=3600)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'Failed to fetch sparkline: {str(e)}'}), 503


@app.route('/api/config', methods=['GET'])
@handle_errors
def get_configuration():
    """Get current configuration."""
    config = get_config()
    return jsonify({
        'start_date': config.app_config.start_date,
        'end_date': config.app_config.end_date,
        'train_test_split': config.app_config.train_test_split,
        'batch_size': config.app_config.batch_size,
        'gru_epochs': config.app_config.gru_epochs,
        'lstm_epochs': config.app_config.lstm_epochs,
        'transformer_epochs': config.app_config.transformer_epochs,
        'hidden_dim': config.app_config.hidden_dim,
        'num_layers': config.app_config.num_layers,
        'learning_rate': config.app_config.learning_rate,
        'transformer_learning_rate': config.app_config.transformer_learning_rate,
        'price_lag_days': config.app_config.price_lag_days,
        'volume_lag_days': config.app_config.volume_lag_days,
        'fgi_lag_days': config.app_config.fgi_lag_days,
        'market_cap_lag_days': config.app_config.market_cap_lag_days,
        'rsi_window': config.app_config.rsi_window,
        'macd_fast': config.app_config.macd_fast,
        'macd_slow': config.app_config.macd_slow,
        'macd_signal': config.app_config.macd_signal,
        'output_dir': config.app_config.output_dir
    })


@app.route('/api/config', methods=['PUT'])
@handle_errors
def update_configuration():
    """Update configuration."""
    config = get_config()
    data = request.json
    
    # Date range settings
    if 'start_date' in data:
        config.app_config.start_date = data['start_date']
    if 'end_date' in data:
        config.app_config.end_date = data['end_date']
    
    # Model settings
    if 'train_test_split' in data:
        config.app_config.train_test_split = float(data['train_test_split'])
    if 'batch_size' in data:
        config.app_config.batch_size = int(data['batch_size'])
    if 'gru_epochs' in data:
        config.app_config.gru_epochs = int(data['gru_epochs'])
    if 'lstm_epochs' in data:
        config.app_config.lstm_epochs = int(data['lstm_epochs'])
    if 'transformer_epochs' in data:
        config.app_config.transformer_epochs = int(data['transformer_epochs'])
    if 'hidden_dim' in data:
        config.app_config.hidden_dim = int(data['hidden_dim'])
    if 'num_layers' in data:
        config.app_config.num_layers = int(data['num_layers'])
    if 'learning_rate' in data:
        config.app_config.learning_rate = float(data['learning_rate'])
    if 'transformer_learning_rate' in data:
        config.app_config.transformer_learning_rate = float(data['transformer_learning_rate'])
    
    config.save_config()
    return jsonify({'message': 'Configuration updated successfully'})


@app.route('/api/data/fetch', methods=['POST'])
@handle_errors
def fetch_data():
    # Accept temporary overrides for feature engineering / indicators and persist
    config = get_config()
    appcfg = config.app_config

    # Feature engineering / indicator overrides
    if 'price_lag_days' in request.json:
        try:
            appcfg.price_lag_days = int(request.json.get('price_lag_days'))
        except Exception:
            pass
    if 'volume_lag_days' in request.json:
        try:
            appcfg.volume_lag_days = int(request.json.get('volume_lag_days'))
        except Exception:
            pass
    if 'fgi_lag_days' in request.json:
        try:
            appcfg.fgi_lag_days = int(request.json.get('fgi_lag_days'))
        except Exception:
            pass
    if 'market_cap_lag_days' in request.json:
        try:
            appcfg.market_cap_lag_days = int(request.json.get('market_cap_lag_days'))
        except Exception:
            pass
    if 'rsi_window' in request.json:
        try:
            appcfg.rsi_window = int(request.json.get('rsi_window'))
        except Exception:
            pass
    if 'macd_fast' in request.json:
        try:
            appcfg.macd_fast = int(request.json.get('macd_fast'))
        except Exception:
            pass
    if 'macd_slow' in request.json:
        try:
            appcfg.macd_slow = int(request.json.get('macd_slow'))
        except Exception:
            pass
    if 'macd_signal' in request.json:
        try:
            appcfg.macd_signal = int(request.json.get('macd_signal'))
        except Exception:
            pass

    # Other training-related overrides
    if 'train_test_split' in request.json:
        try:
            appcfg.train_test_split = float(request.json.get('train_test_split'))
        except Exception:
            pass
    if 'num_layers' in request.json:
        try:
            appcfg.num_layers = int(request.json.get('num_layers'))
        except Exception:
            pass
    if 'transformer_learning_rate' in request.json:
        try:
            appcfg.transformer_learning_rate = float(request.json.get('transformer_learning_rate'))
        except Exception:
            pass

    # Persist any changes
    try:
        logger.info("Received configuration overrides for data fetch: %s", request.json if request and hasattr(request, 'json') else {})
        config.save_config()
    except Exception:
        pass
    """Fetch and prepare data for a specific cryptocurrency."""
    service = get_service()
    
    data_json = request.json or {}
    symbol = data_json.get('symbol', '').upper() if data_json.get('symbol') else None
    force_refresh = data_json.get('force_refresh', False)
    
    if not symbol:
        return jsonify({'error': 'Symbol is required'}), 400
    
    data = service.fetch_and_prepare_data(
        target_symbol=symbol,
        force_refresh=force_refresh
    )
    
    return jsonify({
        'message': f'Data fetched successfully for {symbol}',
        'symbol': symbol,
        'shape': list(data.shape),
        'date_range': {
            'start': data.index[0].isoformat(),
            'end': data.index[-1].isoformat()
        },
        'columns': len(data.columns)
    })


@app.route('/api/train', methods=['POST'])
@handle_errors
def train_models():
    """Train models for a cryptocurrency."""
    global training_status
    
    data = request.json
    if not data or 'symbol' not in data:
        return jsonify({'error': 'Symbol is required'}), 400
    
    symbol = data['symbol'].upper()
    model_types = data.get('model_types')  # Optional: train specific models
    # Allow frontend to pass training/configuration options
    config = get_config()
    # Generic overrides
    if 'batch_size' in data:
        try:
            config.app_config.batch_size = int(data.get('batch_size'))
        except Exception:
            pass
    if 'learning_rate' in data:
        try:
            config.app_config.learning_rate = float(data.get('learning_rate'))
        except Exception:
            pass
    if 'hidden_dim' in data:
        try:
            config.app_config.hidden_dim = int(data.get('hidden_dim'))
        except Exception:
            pass
    if 'start_date' in data:
        config.app_config.start_date = data.get('start_date')
    if 'end_date' in data:
        config.app_config.end_date = data.get('end_date')

    # Epochs: allow generic 'epochs' to set all, or allow model-specific overrides
    if 'epochs' in data:
        try:
            e = int(data.get('epochs'))
            config.app_config.gru_epochs = e
            config.app_config.lstm_epochs = e
            config.app_config.transformer_epochs = e
        except Exception:
            pass
    if 'gru_epochs' in data:
        try:
            config.app_config.gru_epochs = int(data.get('gru_epochs'))
        except Exception:
            pass
        # Log incoming configuration data
        try:
            logger.info("Received configuration overrides for training: %s", data)
        except Exception:
            logger.info("Received configuration overrides for training (unserializable payload)")
    if 'lstm_epochs' in data:
        try:
            config.app_config.lstm_epochs = int(data.get('lstm_epochs'))
        except Exception:
            pass
    if 'transformer_epochs' in data:
        try:
            config.app_config.transformer_epochs = int(data.get('transformer_epochs'))
        except Exception:
            pass

    # Persist updated config so subsequent calls use it
    try:
        config.save_config()
    except Exception:
        pass
    
    training_status = {
        'is_training': True,
        'current_model': symbol,
        'progress': 0,
        'message': f'Training models for {symbol}...'
    }
    
    try:
        service = get_service()
        
        # Fetch data if not already done
        training_status['message'] = 'Fetching data...'
        training_status['progress'] = 10
        service.fetch_and_prepare_data()
        
        # Train models
        training_status['message'] = 'Training models...'
        training_status['progress'] = 30
        result = service.train_models(symbol, model_types)
        
        training_status['progress'] = 100
        training_status['message'] = 'Training complete!'
        training_status['is_training'] = False
        
        return jsonify({
            'message': f'Models trained successfully for {symbol}',
            'result': result
        })
        
    except Exception as e:
        training_status['is_training'] = False
        training_status['message'] = f'Training failed: {str(e)}'
        raise


@app.route('/api/train/status', methods=['GET'])
def get_training_status():
    """Get current training status."""
    return jsonify(training_status)


@app.route('/api/train/async', methods=['POST'])
@handle_errors
def train_models_async():
    """Train models asynchronously with progress tracking."""
    data = request.json
    if not data or 'symbol' not in data:
        return jsonify({'error': 'Symbol is required'}), 400
    
    symbol = data['symbol'].upper()
    model_types = data.get('model_types')
    session_id = str(uuid.uuid4())
    
    # Store session info
    training_sessions[session_id] = {
        'symbol': symbol,
        'model_types': model_types,
        'status': 'pending',
        'started_at': datetime.now().isoformat()
    }
    
    def train_task(progress_callback=None):
        """Background training task."""
        service = get_service()
        
        # Fetch data
        if progress_callback:
            progress_callback(10, 'Fetching market data...')
        service.fetch_and_prepare_data(target_symbol=symbol)
        
        # Train models
        if progress_callback:
            progress_callback(30, 'Training models...')
        
        result = service.train_models(symbol, model_types)
        
        training_sessions[session_id]['status'] = 'completed'
        training_sessions[session_id]['result'] = result
        
        return result
    
    # Submit task
    task_id = submit_task(train_task)
    training_sessions[session_id]['task_id'] = task_id
    
    return jsonify({
        'session_id': session_id,
        'task_id': task_id,
        'message': f'Training started for {symbol}',
        'status': 'pending'
    })


@app.route('/api/train/async/<session_id>', methods=['GET'])
@handle_errors
def get_async_training_status(session_id: str):
    """Get status of async training session."""
    if session_id not in training_sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    session = training_sessions[session_id]
    task_id = session.get('task_id')
    
    if task_id:
        task_result = get_task_status(task_id)
        if task_result:
            session['progress'] = task_result.progress
            session['progress_message'] = task_result.progress_message
            session['status'] = task_result.status.value
            
            if task_result.error:
                session['error'] = task_result.error
            if task_result.result:
                session['result'] = task_result.result
    
    return jsonify(session)


@app.route('/api/train/cancel/<session_id>', methods=['POST'])
@handle_errors
def cancel_training(session_id: str):
    """Cancel an async training session."""
    if session_id not in training_sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    session = training_sessions[session_id]
    task_id = session.get('task_id')
    
    if task_id:
        cancelled = cancel_task(task_id)
        if cancelled:
            session['status'] = 'cancelled'
            return jsonify({'message': 'Training cancelled', 'session_id': session_id})
    
    return jsonify({'error': 'Cannot cancel running task'}), 400


@app.route('/api/predict', methods=['POST'])
@handle_errors
def get_predictions():
    """Get predictions for a cryptocurrency."""
    data = request.json
    if not data or 'symbol' not in data:
        return jsonify({'error': 'Symbol is required'}), 400
    
    symbol = data['symbol'].upper()
    include_confidence = data.get('include_confidence', True)
    
    service = get_service()
    predictions = service.get_predictions(symbol, include_confidence=include_confidence)
    
    # Convert numpy arrays to lists for JSON serialization
    result = {
        'symbol': symbol,
        'dates': [d.isoformat() if hasattr(d, 'isoformat') else str(d) for d in predictions['dates']],
        'actual': predictions['actual'].tolist(),
        'predictions': {
            model: pred.tolist() 
            for model, pred in predictions['predictions'].items()
        }
    }
    
    if include_confidence and 'confidence_intervals' in predictions:
        result['confidence_intervals'] = {
            model: {
                'lower': ci['lower'].tolist(),
                'upper': ci['upper'].tolist(),
                'std_dev': float(ci['std_dev'])
            }
            for model, ci in predictions['confidence_intervals'].items()
        }
    
    return jsonify(result)


@app.route('/api/predict/future', methods=['POST'])
@handle_errors
def get_future_predictions():
    """Get predictions including future days."""
    data = request.json
    if not data or 'symbol' not in data:
        return jsonify({'error': 'Symbol is required'}), 400
    
    symbol = data['symbol'].upper()
    days_ahead = data.get('days_ahead', 7)
    try:
        days_ahead = int(days_ahead)
    except Exception:
        days_ahead = 7

    # Validate days_ahead is within allowed range (1-30 days)
    if days_ahead < 1:
        days_ahead = 1
    elif days_ahead > 30:
        days_ahead = 30
    
    service = get_service()
    predictions_df, current_price = service.get_future_predictions(symbol, days_ahead)
    
    # Convert DataFrame to records, replacing NaN with None for valid JSON
    predictions_records = predictions_df.where(predictions_df.notna(), None).to_dict(orient='records')
    
    # Ensure dates are in ISO format string (same as /api/predict)
    for record in predictions_records:
        if 'date' in record and hasattr(record['date'], 'isoformat'):
            record['date'] = record['date'].isoformat()
    
    return jsonify({
        'symbol': symbol,
        'current_price': current_price,
        'predictions': predictions_records
    })


@app.route('/api/evaluate', methods=['POST'])
@handle_errors
def evaluate_models():
    """Evaluate model performance."""
    data = request.json
    if not data or 'symbol' not in data:
        return jsonify({'error': 'Symbol is required'}), 400
    
    symbol = data['symbol'].upper()
    
    service = get_service()
    metrics = service.evaluate_models(symbol)
    
    return jsonify({
        'symbol': symbol,
        'metrics': metrics.to_dict(orient='index')
    })


@app.route('/api/btc-dominance', methods=['GET'])
@handle_errors
def get_btc_dominance():
    """Get current BTC dominance."""
    service = get_service()
    dominance = service.get_btc_dominance()
    
    if dominance is None:
        return jsonify({'error': 'Could not fetch BTC dominance'}), 503
    
    return jsonify({
        'btc_dominance': dominance,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/models/save', methods=['POST'])
@handle_errors
def save_models():
    """Save trained models to disk."""
    service = get_service()
    path = service.save_models()
    
    return jsonify({
        'message': 'Models saved successfully',
        'path': path
    })


@app.route('/api/models/load', methods=['POST'])
@handle_errors
def load_models():
    """Load trained models from disk."""
    data = request.json or {}
    input_dim = data.get('input_dim')
    
    service = get_service()
    service.load_models(input_dim)
    
    return jsonify({
        'message': 'Models loaded successfully',
        'models': list(service.model_manager.models.keys())
    })


@app.route('/api/export', methods=['POST'])
@handle_errors
def export_results():
    """Export prediction results to CSV."""
    data = request.json
    if not data or 'symbol' not in data:
        return jsonify({'error': 'Symbol is required'}), 400
    
    symbol = data['symbol'].upper()
    filename = data.get('filename')
    
    service = get_service()
    output_path = service.save_results(symbol, filename)
    
    return jsonify({
        'message': 'Results exported successfully',
        'path': output_path
    })


# ============================================
# Error Handlers
# ============================================

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500


# ============================================
# Run the app
# ============================================

def create_app():
    init_cache(redis_host=os.getenv("REDIS_HOST"))
    return app

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='CryptoPredictor API Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    logger.info(f"Starting CryptoPredictor API on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)
