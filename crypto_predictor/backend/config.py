"""
Configuration management for CryptoPredictor.

The predictor is a *direction classifier*: for every supported timeframe
(1h, 4h, 1d) it estimates whether the next candle will close up, down or
roughly flat (neutral) relative to the current close.
"""

import json
import logging
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ============================================
# Timeframes
# ============================================

@dataclass(frozen=True)
class TimeframeSpec:
    """Static properties of a candle timeframe."""
    name: str
    minutes: int
    history_days: int          # how much history to download by default
    neutral_threshold: float   # |log return| below this is labelled "neutral"
    window: int                # sequence length (in candles) for GRU/LSTM/Transformer

    @property
    def millis(self) -> int:
        return self.minutes * 60 * 1000

    @property
    def bars_per_year(self) -> float:
        return 365 * 24 * 60 / self.minutes


TIMEFRAMES: Dict[str, TimeframeSpec] = {
    '1h': TimeframeSpec('1h', 60, history_days=730, neutral_threshold=0.0025, window=48),
    '4h': TimeframeSpec('4h', 240, history_days=2190, neutral_threshold=0.005, window=42),
    '1d': TimeframeSpec('1d', 1440, history_days=3000, neutral_threshold=0.01, window=30),
}


def get_timeframe(name: str) -> TimeframeSpec:
    if name not in TIMEFRAMES:
        raise ValueError(f"Unsupported timeframe '{name}'. Supported: {list(TIMEFRAMES)}")
    return TIMEFRAMES[name]


# ============================================
# Assets
# ============================================

@dataclass
class CryptoConfig:
    """Configuration for a single cryptocurrency."""
    symbol: str
    name: str
    binance_symbol: str
    yf_ticker: str
    enabled: bool = True


# Stablecoins (e.g. USDT) are intentionally not listed: there is no direction to predict.
DEFAULT_CRYPTOCURRENCIES: List[CryptoConfig] = [
    CryptoConfig("BTC", "Bitcoin", "BTCUSDT", "BTC-USD"),
    CryptoConfig("ETH", "Ethereum", "ETHUSDT", "ETH-USD"),
    CryptoConfig("XRP", "Ripple", "XRPUSDT", "XRP-USD"),
    CryptoConfig("SOL", "Solana", "SOLUSDT", "SOL-USD"),
    CryptoConfig("BNB", "Binance Coin", "BNBUSDT", "BNB-USD"),
    CryptoConfig("VTHO", "VeThor Token", "VTHOUSDT", "VTHO-USD"),
    CryptoConfig("SEI", "Sei", "SEIUSDT", "SEI-USD"),
    CryptoConfig("APT", "Aptos", "APTUSDT", "APT21794-USD"),
    CryptoConfig("FET", "Fetch.ai", "FETUSDT", "FET-USD"),
    CryptoConfig("NEAR", "NEAR Protocol", "NEARUSDT", "NEAR-USD"),
    CryptoConfig("RENDER", "Render Token", "RENDERUSDT", "RENDER-USD"),
    CryptoConfig("GALA", "Gala", "GALAUSDT", "GALA-USD"),
    CryptoConfig("ARB", "Arbitrum", "ARBUSDT", "ARB11841-USD"),
    CryptoConfig("ZEC", "Zcash", "ZECUSDT", "ZEC-USD"),
    CryptoConfig("IMX", "Immutable X", "IMXUSDT", "IMX-USD"),
    CryptoConfig("LINK", "Chainlink", "LINKUSDT", "LINK-USD"),
]

# Daily macro context (fetched from Yahoo Finance when available).
MARKET_INDICES: Dict[str, str] = {
    "^GSPC": "SP500",
    "^IXIC": "NASDAQ",
    "DX-Y.NYB": "DXY",
    "^VIX": "VIX",
}

MODEL_NAMES = ['gbm', 'logreg', 'gru', 'lstm', 'transformer']
DEFAULT_MODELS = ['gbm', 'logreg', 'gru']


@dataclass
class AppConfig:
    """User-tunable application settings."""

    # Data
    data_source: str = 'auto'              # 'auto' (Binance, then Yahoo), 'binance', 'yfinance'
    use_btc_context: bool = True           # add BTC features when predicting an altcoin
    use_macro: bool = True                 # add daily stock-index / Fear & Greed features

    # Labels
    label_mode: str = 'ternary'            # 'ternary' (down/neutral/up) or 'binary' (down/up)
    horizon_bars: int = 1                  # predict the move over the next N candles
    neutral_thresholds: Dict[str, float] = field(
        default_factory=lambda: {k: v.neutral_threshold for k, v in TIMEFRAMES.items()}
    )

    # Features
    return_lags: int = 6
    rsi_window: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    atr_window: int = 14
    bb_window: int = 20

    # Evaluation (walk-forward)
    walk_forward_splits: int = 4
    initial_train_frac: float = 0.6
    val_frac: float = 0.15                 # tail of each training fold used for early stopping

    # Neural network models
    hidden_dim: int = 64
    num_layers: int = 1
    dropout: float = 0.2
    epochs: int = 40
    patience: int = 6
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    windows: Dict[str, int] = field(
        default_factory=lambda: {k: v.window for k, v in TIMEFRAMES.items()}
    )

    # Gradient boosting
    gbm_estimators: int = 400
    gbm_learning_rate: float = 0.03
    gbm_num_leaves: int = 15

    # Backtest
    fee_bps: float = 10.0                  # per side, in basis points (10 = 0.1%)
    min_confidence: float = 0.0            # trade only when P(direction) >= this
    allow_short: bool = True

    seed: int = 42
    output_dir: str = field(default_factory=lambda: str(Path.home() / "crypto_predictor_output"))

    def threshold_for(self, timeframe: str) -> float:
        return float(self.neutral_thresholds.get(timeframe, get_timeframe(timeframe).neutral_threshold))

    def window_for(self, timeframe: str) -> int:
        return int(self.windows.get(timeframe, get_timeframe(timeframe).window))

    def update(self, values: Dict) -> None:
        """Apply a partial update, coercing each value to the field's current type."""
        known = {f.name for f in fields(self)}
        for key, value in (values or {}).items():
            if key not in known or value is None:
                continue
            current = getattr(self, key)
            try:
                if isinstance(current, bool):
                    value = value if isinstance(value, bool) else str(value).lower() in ('1', 'true', 'yes', 'on')
                elif isinstance(current, int):
                    value = int(value)
                elif isinstance(current, float):
                    value = float(value)
                elif isinstance(current, dict):
                    if not isinstance(value, dict):
                        continue
                    merged = dict(current)
                    merged.update({k: type(next(iter(current.values()), 0.0))(v) for k, v in value.items()
                                   if k in TIMEFRAMES})
                    value = merged
                else:
                    value = str(value)
            except (TypeError, ValueError):
                raise ValueError(f"Invalid value for '{key}': {value!r}")
            setattr(self, key, value)
        self.validate()

    def validate(self) -> None:
        if self.data_source not in ('auto', 'binance', 'yfinance'):
            raise ValueError("data_source must be 'auto', 'binance' or 'yfinance'")
        if self.label_mode not in ('ternary', 'binary'):
            raise ValueError("label_mode must be 'ternary' or 'binary'")
        if not 1 <= self.horizon_bars <= 24:
            raise ValueError("horizon_bars must be between 1 and 24")
        if not 2 <= self.walk_forward_splits <= 10:
            raise ValueError("walk_forward_splits must be between 2 and 10")
        if not 0.3 <= self.initial_train_frac <= 0.9:
            raise ValueError("initial_train_frac must be between 0.3 and 0.9")
        if not 0.0 <= self.min_confidence < 1.0:
            raise ValueError("min_confidence must be in [0, 1)")


class ConfigManager:
    """Manages application configuration with persistence."""

    CONFIG_FILENAME = "config_v2.json"  # v1 (price regression) files are left untouched

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._get_default_config_path()
        self.app_config = AppConfig()
        self.cryptocurrencies = [CryptoConfig(**asdict(c)) for c in DEFAULT_CRYPTOCURRENCIES]
        self._load_config()

    @classmethod
    def _get_default_config_path(cls) -> str:
        config_dir = Path.home() / ".crypto_predictor"
        config_dir.mkdir(parents=True, exist_ok=True)
        return str(config_dir / cls.CONFIG_FILENAME)

    def _load_config(self) -> None:
        try:
            path = Path(self.config_path)
            if path.exists():
                data = json.loads(path.read_text())
                self.app_config.update(data.get('app_config', {}))
        except Exception as e:
            logger.warning("Could not load config from %s: %s", self.config_path, e)
            self.app_config = AppConfig()

    def save_config(self) -> None:
        try:
            Path(self.config_path).write_text(json.dumps({'app_config': asdict(self.app_config)}, indent=2))
        except Exception as e:
            logger.warning("Could not save config to %s: %s", self.config_path, e)

    def get_enabled_cryptos(self) -> List[CryptoConfig]:
        return [c for c in self.cryptocurrencies if c.enabled]

    def get_crypto(self, symbol: str) -> CryptoConfig:
        for c in self.cryptocurrencies:
            if c.symbol == symbol.upper():
                return c
        raise ValueError(f"Unknown symbol '{symbol}'. Valid symbols: {[c.symbol for c in self.cryptocurrencies]}")

    def get_output_path(self, *parts: str) -> str:
        path = Path(self.app_config.output_dir).joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)


_config_manager: Optional[ConfigManager] = None


def get_config() -> ConfigManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def reset_config() -> None:
    """Reset the global configuration manager."""
    global _config_manager
    _config_manager = None
