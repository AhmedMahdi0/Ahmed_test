"""
Base configuration classes for multi-asset support.
Allows the system to handle both cryptocurrencies and stocks.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional


class AssetType(Enum):
    """Types of assets supported by the predictor."""
    CRYPTO = "crypto"
    STOCK = "stock"
    FOREX = "forex"
    COMMODITY = "commodity"


@dataclass
class BaseAssetConfig:
    """Base configuration for any tradeable asset."""
    symbol: str
    ticker: str
    name: str
    asset_type: AssetType
    enabled: bool = True
    exchange: Optional[str] = None
    currency: str = "USD"
    
    @property
    def display_name(self) -> str:
        """Get display name for UI."""
        return f"{self.name} ({self.symbol})"
    
    @property
    def price_column(self) -> str:
        """Get the price column name for this asset."""
        return f"{self.symbol}_Price"
    
    @property
    def volume_column(self) -> str:
        """Get the volume column name for this asset."""
        return f"{self.symbol}_Volume"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            'symbol': self.symbol,
            'ticker': self.ticker,
            'name': self.name,
            'asset_type': self.asset_type.value,
            'enabled': self.enabled,
            'exchange': self.exchange,
            'currency': self.currency
        }
