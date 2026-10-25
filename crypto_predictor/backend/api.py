"""
Flask API backend for CryptoPredictor.

Main flow:
    POST /api/analyze                 -> start a background analysis, returns task_id
    GET  /api/analyze/<task_id>       -> progress / result
    POST /api/analyze/<task_id>/cancel
    GET  /api/signal/<symbol>         -> refresh the next-candle signal with trained models
    GET  /api/export/<symbol>/<tf>    -> CSV of out-of-sample predictions
"""

import gzip
import logging
import os
from datetime import datetime
from functools import wraps
from io import BytesIO
from typing import Optional

from flask import Flask, jsonify, request, send_from_directory, Response
from flask_cors import CORS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from dataclasses import asdict

from .config import get_config, TIMEFRAMES, DEFAULT_MODELS
from .models import MODEL_INFO, available_models, TORCH_AVAILABLE, LIGHTGBM_AVAILABLE
from .prediction_service import PredictionService
from .cache import init_cache
from .tasks import submit_task, get_task_status, cancel_task

_static_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'frontend')
app = Flask(__name__, static_folder=_static_folder, static_url_path='')
CORS(app)

socketio = None
try:
    from .websocket import init_socketio
    socketio = init_socketio(app)
except ImportError:
    logger.warning("WebSocket support not available (install flask-socketio)")

prediction_service: Optional[PredictionService] = None


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
        if 'gzip' not in request.headers.get('Accept-Encoding', ''):
            return response
        if isinstance(response, Response):
            data = response.get_data()
            if len(data) < 500:
                return response
            gzip_buffer = BytesIO()
            with gzip.GzipFile(mode='wb', fileobj=gzip_buffer, compresslevel=6) as gz:
                gz.write(data)
            compressed = gzip_buffer.getvalue()
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


def _parse_timeframes(value) -> list:
    if not value:
        return list(TIMEFRAMES)
    if isinstance(value, str):
        value = [v.strip() for v in value.split(',') if v.strip()]
    bad = [tf for tf in value if tf not in TIMEFRAMES]
    if bad:
        raise ValueError(f"Unsupported timeframes {bad}. Supported: {list(TIMEFRAMES)}")
    return list(dict.fromkeys(value))


# ============================================
# Static file serving (for SPA)
# ============================================

@app.route('/')
def serve_frontend():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    if os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')


# ============================================
# Metadata
# ============================================

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '3.0.0',
        'features': {
            'websocket': socketio is not None,
            'torch': TORCH_AVAILABLE,
            'lightgbm': LIGHTGBM_AVAILABLE,
        }
    })


@app.route('/api/cryptocurrencies', methods=['GET'])
@handle_errors
def get_cryptocurrencies():
    cryptos = get_service().get_available_cryptos()
    return jsonify({'cryptocurrencies': cryptos, 'count': len(cryptos)})


@app.route('/api/timeframes', methods=['GET'])
def get_timeframes():
    cfg = get_config().app_config
    return jsonify({'timeframes': [
        {'name': tf, 'minutes': spec.minutes, 'history_days': spec.history_days,
         'neutral_threshold': cfg.threshold_for(tf), 'window': cfg.window_for(tf)}
        for tf, spec in TIMEFRAMES.items()
    ]})


@app.route('/api/models', methods=['GET'])
def get_models():
    avail = available_models()
    return jsonify({'models': [
        {'name': name, **info, 'available': avail[name], 'default': name in DEFAULT_MODELS}
        for name, info in MODEL_INFO.items()
    ]})


@app.route('/api/config', methods=['GET'])
def get_configuration():
    return jsonify(asdict(get_config().app_config))


@app.route('/api/config', methods=['PUT'])
@handle_errors
def update_configuration():
    config = get_config()
    config.app_config.update(request.get_json(silent=True) or {})
    config.save_config()
    return jsonify({'message': 'Configuration updated', 'config': asdict(config.app_config)})


@app.route('/api/asset/<symbol>/price', methods=['GET'])
@handle_errors
def get_asset_price(symbol: str):
    price = get_service().fetcher.get_latest_price(symbol)
    if price is None:
        return jsonify({'error': 'Price unavailable'}), 503
    return jsonify({'symbol': symbol.upper(), 'price': price, 'timestamp': datetime.now().isoformat()})


# ============================================
# Analysis
# ============================================

@app.route('/api/analyze', methods=['POST'])
@handle_errors
def start_analysis():
    data = request.get_json(silent=True) or {}
    symbol = (data.get('symbol') or '').upper()
    if not symbol:
        raise ValueError('Symbol is required')
    service = get_service()
    service.config.get_crypto(symbol)
    timeframes = _parse_timeframes(data.get('timeframes'))
    models = service._validate_models(data.get('models') or DEFAULT_MODELS)

    if data.get('settings'):
        config = get_config()
        config.app_config.update(data['settings'])
        config.save_config()

    def task(progress_callback=None, cancel_event=None):
        return service.analyze(symbol, timeframes, models,
                               progress_callback=progress_callback, cancel_event=cancel_event)

    task_id = submit_task(task)
    return jsonify({'task_id': task_id, 'symbol': symbol, 'timeframes': timeframes,
                    'models': models, 'status': 'pending'}), 202


@app.route('/api/analyze/<task_id>', methods=['GET'])
@gzip_response
@handle_errors
def analysis_status(task_id: str):
    status = get_task_status(task_id)
    if status is None:
        return jsonify({'error': 'Task not found'}), 404
    return jsonify(status.to_dict())


@app.route('/api/analyze/<task_id>/cancel', methods=['POST'])
@handle_errors
def cancel_analysis(task_id: str):
    if cancel_task(task_id):
        return jsonify({'message': 'Cancellation requested', 'task_id': task_id})
    return jsonify({'error': 'Task is not running'}), 400


@app.route('/api/signal/<symbol>', methods=['GET'])
@handle_errors
def get_signal(symbol: str):
    timeframes = _parse_timeframes(request.args.get('timeframes'))
    return jsonify(get_service().latest_signal(symbol, timeframes))


@app.route('/api/results/<symbol>/<timeframe>', methods=['GET'])
@gzip_response
@handle_errors
def get_results(symbol: str, timeframe: str):
    result = get_service().get_result(symbol, timeframe)
    if result is None:
        return jsonify({'error': f'No analysis for {symbol.upper()} {timeframe} yet'}), 404
    return jsonify(result)


@app.route('/api/export/<symbol>/<timeframe>', methods=['GET'])
@handle_errors
def export_results(symbol: str, timeframe: str):
    csv = get_service().export_csv(symbol, timeframe)
    filename = f"{symbol.upper()}_{timeframe}_oos_predictions.csv"
    return Response(csv, mimetype='text/csv',
                    headers={'Content-Disposition': f'attachment; filename={filename}'})


# ============================================
# Error Handlers
# ============================================

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500


def create_app():
    init_cache(redis_host=os.getenv("REDIS_HOST"))
    return app
