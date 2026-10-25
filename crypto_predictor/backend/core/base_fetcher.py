"""
Base data fetcher interface for multi-asset support.
Provides abstract base class for fetching different asset types.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from datetime import datetime
import pandas as pd

from .base_config import BaseAssetConfig, AssetType


class BaseDataFetcher(ABC):
    """Abstract base class for data fetchers."""
    
    @property
    @abstractmethod
    def asset_type(self) -> AssetType:
        """Return the asset type this fetcher handles."""
        pass
    
    @abstractmethod
    def fetch_price_data(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetch price and volume data for given symbols.
        
        Args:
            symbols: List of asset symbols to fetch.
            start_date: Start date string (YYYY-MM-DD).
            end_date: End date string (YYYY-MM-DD).
            
        Returns:
            DataFrame with price and volume columns for each symbol.
        """
        pass
    
    @abstractmethod
    def get_available_assets(self) -> List[BaseAssetConfig]:
        """
        Get list of available assets for this fetcher.
        
        Returns:
            List of asset configurations.
        """
        pass
    
    @abstractmethod
    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current/latest price for a symbol.
        
        Args:
            symbol: Asset symbol.
            
        Returns:
            Current price or None if unavailable.
        """
        pass
    
    def get_sparkline_data(
        self,
        symbol: str,
        days: int = 7
    ) -> Optional[List[float]]:
        """
        Get sparkline data (price history) for visualization.
        
        Args:
            symbol: Asset symbol.
            days: Number of days of history.
            
        Returns:
            List of prices for sparkline or None if unavailable.
        """
        try:
            from datetime import timedelta
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            
            data = self.fetch_price_data([symbol], start_date, end_date)
            price_col = f"{symbol}_Price"
            
            if price_col in data.columns:
                return data[price_col].dropna().tolist()
            return None
        except Exception:
            return None
    
    def get_market_stats(self, symbol: str) -> Dict[str, Any]:
        """
        Get market statistics for a symbol.
        
        Args:
            symbol: Asset symbol.
            
        Returns:
            Dictionary with market stats (24h change, volume, etc.)
        """
        return {
            'price': self.get_current_price(symbol),
            'change_24h': None,
            'volume_24h': None,
            'market_cap': None
        }
