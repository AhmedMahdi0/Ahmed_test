"""
Configuration management for CryptoPredictor.
Handles dynamic cryptocurrency selection and application settings.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class CryptoConfig:
    """Configuration for a single cryptocurrency."""
    symbol: str
    ticker: str
    name: str
    enabled: bool = True


@dataclass
class AppConfig:
    """Main application configuration."""
    
    # Date range settings
    start_date: str = "2018-02-01"
    end_date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    
    # Model settings
    train_test_split: float = 0.8
    batch_size: int = 64
    gru_epochs: int = 200
    lstm_epochs: int = 200
    transformer_epochs: int = 200
    hidden_dim: int = 64
    num_layers: int = 2
    learning_rate: float = 0.001
    transformer_learning_rate: float = 0.0005
    
    # Lag features
    price_lag_days: int = 7
    volume_lag_days: int = 1
    fgi_lag_days: int = 7
    market_cap_lag_days: int = 1
    
    # Technical indicators
    rsi_window: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # Output settings - Cross-platform compatible
    output_dir: str = field(default_factory=lambda: str(Path.home() / "crypto_predictor_output"))
    
    # API rate limiting
    api_delay_seconds: float = 1.2

# Available cryptocurrencies
DEFAULT_CRYPTOCURRENCIES: List[CryptoConfig] = [
    CryptoConfig("BTC", "BTC-USD", "Bitcoin", enabled=True),
    CryptoConfig("ETH", "ETH-USD", "Ethereum", enabled=True),
    CryptoConfig("XRP", "XRP-USD", "Ripple", enabled=True),
    CryptoConfig("SOL", "SOL-USD", "Solana", enabled=True),
    CryptoConfig("USDT", "USDT-USD", "Tether", enabled=True),
    CryptoConfig("BNB", "BNB-USD", "Binance Coin", enabled=True),
    CryptoConfig("VTHO", "VTHO-USD", "VeThor Token", enabled=True),
    CryptoConfig("SEI", "SEI-USD", "Sei", enabled=True),
    CryptoConfig("APT", "APT21794-USD", "Aptos", enabled=True),
    CryptoConfig("FET", "FET-USD", "Fetch.ai", enabled=True),
    CryptoConfig("NEAR", "NEAR-USD", "NEAR Protocol", enabled=True),
    CryptoConfig("RENDER", "RENDER-USD", "Render Token", enabled=True),
    CryptoConfig("GALA", "GALA-USD", "Gala", enabled=True),
    CryptoConfig("ARB", "ARB11841-USD", "Arbitrum", enabled=True),
    CryptoConfig("ZEC", "ZEC-USD", "Zcash", enabled=True),
    CryptoConfig("IMX", "IMX-USD", "Immutable X", enabled=True),
    CryptoConfig("LINK", "LINK-USD", "Chainlink", enabled=True),
]

# Global market indices
MARKET_INDICES: Dict[str, str] = {
    "^GSPC": "SP500",
    "^IXIC": "NASDAQ",
    "^DJI": "DJIA",
    "^FTSE": "FTSE100",
    "^GDAXI": "DAX",
    "^N225": "Nikkei",
    "000001.SS": "Shanghai",
    "^HSI": "HangSeng",
    "DX-Y.NYB": "DXY",
    "^VIX": "VIX",
}

class ConfigManager:
    """Manages application configuration with persistence."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._get_default_config_path()
        self.app_config = AppConfig()
        self.cryptocurrencies = DEFAULT_CRYPTOCURRENCIES.copy()
        self._load_config()
    
    @staticmethod
    def _get_default_config_path() -> str:
        """Get the default config file path (cross-platform)."""
        config_dir = Path.home() / ".crypto_predictor"
        config_dir.mkdir(parents=True, exist_ok=True)
        return str(config_dir / "config.json")
    
    def _load_config(self) -> None:
        """Load configuration from file if it exists."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
                    
                # Update app config
                if 'app_config' in data:
                    for key, value in data['app_config'].items():
                        if hasattr(self.app_config, key):
                            setattr(self.app_config, key, value)
                if 'cryptocurrencies' in data:
                    logger.info("Ignoring persisted 'cryptocurrencies' in %s; using DEFAULT_CRYPTOCURRENCIES instead", self.config_path)
        except Exception as e:
            print(f"Warning: Could not load config from {self.config_path}: {e}")
    
    def save_config(self) -> None:
        """Save current configuration to file."""
        try:
            # Log the current app_config when saving (helpful when frontend sends overrides)
            try:
                logger.info("Received configuration from frontend / saving config: %s", vars(self.app_config))
            except Exception:
                logger.info("Saving configuration (could not serialize AppConfig via vars)")
            data = {
                'app_config': {
                    # Date range
                    'start_date': self.app_config.start_date,
                    'end_date': self.app_config.end_date,
                    # Model training settings
                    'train_test_split': self.app_config.train_test_split,
                    'batch_size': self.app_config.batch_size,
                    'gru_epochs': self.app_config.gru_epochs,
                    'lstm_epochs': self.app_config.lstm_epochs,
                    'transformer_epochs': self.app_config.transformer_epochs,
                    # Model architecture
                    'hidden_dim': self.app_config.hidden_dim,
                    'num_layers': self.app_config.num_layers,
                    'learning_rate': self.app_config.learning_rate,
                    'transformer_learning_rate': self.app_config.transformer_learning_rate,
                    # Feature engineering
                    'price_lag_days': self.app_config.price_lag_days,
                    'volume_lag_days': self.app_config.volume_lag_days,
                    'fgi_lag_days': self.app_config.fgi_lag_days,
                    'market_cap_lag_days': self.app_config.market_cap_lag_days,
                    'rsi_window': self.app_config.rsi_window,
                    'macd_fast': self.app_config.macd_fast,
                    'macd_slow': self.app_config.macd_slow,
                    'macd_signal': self.app_config.macd_signal,
                    # Output
                    'output_dir': self.app_config.output_dir,
                    'api_delay_seconds': self.app_config.api_delay_seconds,
                },
                'cryptocurrencies': [
                    {
                        'symbol': c.symbol,
                        'ticker': c.ticker,
                        'name': c.name,
                        'enabled': c.enabled
                    }
                    for c in self.cryptocurrencies
                ]
            }
            
            with open(self.config_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save config to {self.config_path}: {e}")
    
    def get_enabled_cryptos(self) -> List[CryptoConfig]:
        """Get list of enabled cryptocurrencies."""
        return [c for c in self.cryptocurrencies if c.enabled]
    
    def get_crypto_tickers(self) -> Dict[str, str]:
        """Get dictionary of ticker -> symbol for enabled cryptos."""
        return {c.ticker: c.symbol for c in self.get_enabled_cryptos()}
    
    def set_target_crypto(self, symbol: str) -> None:
        """Set the target cryptocurrency for prediction."""
        # Validate that the symbol exists
        valid_symbols = [c.symbol for c in self.cryptocurrencies]
        if symbol not in valid_symbols:
            raise ValueError(f"Invalid symbol '{symbol}'. Valid symbols: {valid_symbols}")
        self.target_symbol = symbol
    
    def get_output_path(self, filename: str) -> str:
        """Get cross-platform output file path."""
        output_dir = Path(self.app_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return str(output_dir / filename)


# Global config instance
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
