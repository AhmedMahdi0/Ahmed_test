"""
Core module for Asset Predictor.
Contains base classes and abstractions for supporting multiple asset types.
"""

from .base_fetcher import BaseDataFetcher
from .base_config import BaseAssetConfig, AssetType

__all__ = ['BaseDataFetcher', 'BaseAssetConfig', 'AssetType']
