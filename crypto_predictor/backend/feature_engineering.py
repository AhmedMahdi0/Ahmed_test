import logging
from typing import List, Optional
import pandas as pd
import numpy as np

from .config import get_config, MARKET_INDICES

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Creates features for cryptocurrency prediction models."""
    
    def __init__(self):
        self.config = get_config()
    
    def calculate_rsi(self, series: pd.Series, window: int = 30) -> pd.Series:
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
        
        # Avoid division by zero
        rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_macd(self, 
                       series: pd.Series, 
                       fast: int = 12, 
                       slow: int = 26, 
                       signal: int = 9) -> pd.Series:
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        
        return macd_line
    
    def calculate_total_market_cap(self, data: pd.DataFrame) -> pd.Series:
        price_cols = [col for col in data.columns if '_Price' in col]
        return data[price_cols].sum(axis=1)
    
    def add_technical_indicators(self, 
                                  data: pd.DataFrame,
                                  symbols: Optional[List[str]] = None) -> pd.DataFrame:
        result = data.copy()
        
        if symbols is None:
            symbols = [col.replace('_Price', '') for col in data.columns if '_Price' in col]
        
        rsi_columns = []
        macd_columns = []
        
        for symbol in symbols:
            price_col = f'{symbol}_Price'
            if price_col in data.columns:
                # Calculate RSI
                rsi_series = self.calculate_rsi(
                    data[price_col], 
                    window=self.config.app_config.rsi_window
                )
                rsi_series.name = f'RSI_{symbol}'
                rsi_columns.append(rsi_series)
                
                # Calculate MACD
                macd_series = self.calculate_macd(
                    data[price_col],
                    fast=self.config.app_config.macd_fast,
                    slow=self.config.app_config.macd_slow,
                    signal=self.config.app_config.macd_signal
                )
                macd_series.name = f'MACD_{symbol}'
                macd_columns.append(macd_series)
        
        # Concatenate all indicators at once for efficiency
        if rsi_columns:
            result = pd.concat([result] + rsi_columns, axis=1)
        if macd_columns:
            result = pd.concat([result] + macd_columns, axis=1)
        
        logger.info(f"Added technical indicators for {len(symbols)} symbols")
        return result
    
    def add_lag_features(self,
                          data: pd.DataFrame,
                          symbols: Optional[List[str]] = None,
                          price_lag_days: Optional[int] = None,
                          volume_lag_days: Optional[int] = None) -> pd.DataFrame:
        result = data.copy()
        # Use explicit config values when not provided
        price_lag_days = price_lag_days or self.config.app_config.price_lag_days
        volume_lag_days = volume_lag_days or self.config.app_config.volume_lag_days
        
        if symbols is None:
            symbols = [col.replace('_Price', '') for col in data.columns if '_Price' in col]
        
        lag_dfs = []
        
        for symbol in symbols:
            price_col = f'{symbol}_Price'
            volume_col = f'{symbol}_Volume'
            
            if price_col in data.columns and volume_col in data.columns:
                symbol_lags = pd.DataFrame(index=data.index)

                # Price lags
                for lag in range(1, price_lag_days + 1):
                    symbol_lags[f'{symbol}_Price_lag{lag}'] = data[price_col].shift(lag)

                # Volume lags (allow different window)
                for lag in range(1, volume_lag_days + 1):
                    symbol_lags[f'{symbol}_Volume_lag{lag}'] = data[volume_col].shift(lag)

                lag_dfs.append(symbol_lags)
        
        if lag_dfs:
            result = pd.concat([result] + lag_dfs, axis=1)
        
        logger.info(f"Added price lags (last {price_lag_days} days) and volume lags (last {volume_lag_days} days) for {len(symbols)} symbols")
        return result
    
    def add_fgi_lags(self, data: pd.DataFrame, lag_days: int = 7) -> pd.DataFrame:
        result = data.copy()
        
        if 'FGI' not in data.columns:
            logger.warning("FGI column not found, skipping FGI lags")
            return result
        
        fgi_lags = pd.DataFrame(index=data.index)
        for lag in range(1, lag_days + 1):
            fgi_lags[f'FGI_lag{lag}'] = data['FGI'].shift(lag)
        
        result = pd.concat([result, fgi_lags], axis=1)
        logger.info(f"Added {lag_days}-day FGI lag features")
        return result
    
    def add_market_cap_lags(self, data: pd.DataFrame, lag_days: int = 7) -> pd.DataFrame:
        result = data.copy()
        
        if 'Total_Market_Cap' not in data.columns:
            logger.warning("Total_Market_Cap column not found, skipping market cap lags")
            return result
        
        market_cap_lags = pd.DataFrame(index=data.index)
        for lag in range(1, lag_days + 1):
            market_cap_lags[f'Market_Cap_lag{lag}'] = data['Total_Market_Cap'].shift(lag)
        
        result = pd.concat([result, market_cap_lags], axis=1)
        logger.info(f"Added {lag_days}-day market cap lag features")
        return result
    
    def prepare_features(self, 
                          data: pd.DataFrame,
                          target_symbol: str) -> tuple:
        # Market features (common for all cryptos)
        market_cols = list(MARKET_INDICES.values())
        market_cols = [col for col in market_cols if col in data.columns]
        
        # Add target-specific RSI and MACD
        market_cols.extend([f'RSI_{target_symbol}', f'MACD_{target_symbol}'])
        market_cols = [col for col in market_cols if col in data.columns]
        
        # FGI lags
        fgi_lags = [f'FGI_lag{i}' for i in range(1, self.config.app_config.fgi_lag_days + 1)]
        fgi_lags = [col for col in fgi_lags if col in data.columns]
        
        # Price and volume lags for target (volume may use a different lag length)
        price_lags = [f'{target_symbol}_Price_lag{i}' for i in range(1, self.config.app_config.price_lag_days + 1)]
        volume_lags = [f'{target_symbol}_Volume_lag{i}' for i in range(1, self.config.app_config.volume_lag_days + 1)]
        price_lags = [col for col in price_lags if col in data.columns]
        volume_lags = [col for col in volume_lags if col in data.columns]
        
        # Market cap lags
        market_cap_lags = [f'Market_Cap_lag{i}' for i in range(1, self.config.app_config.market_cap_lag_days + 1)]
        market_cap_lags = [col for col in market_cap_lags if col in data.columns]
        
        # Combine all features
        feature_cols = market_cols + fgi_lags + price_lags + volume_lags + market_cap_lags
        
        # Target
        target_col = f'{target_symbol}_Price'
        
        if target_col not in data.columns:
            raise ValueError(f"Target column {target_col} not found in data")
        
        X = data[feature_cols]
        y = data[[target_col]]
        
        logger.info(f"Prepared features for {target_symbol}: X shape={X.shape}, y shape={y.shape}")
        return X, y
    
    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting feature engineering pipeline...")
        
        # Get available symbols
        symbols = [col.replace('_Price', '') for col in data.columns if '_Price' in col]
        
        # Calculate total market cap
        logger.info("Calculating total market cap...")
        data['Total_Market_Cap'] = self.calculate_total_market_cap(data)
        
        # Forward fill market indices
        logger.info("Handling missing values in market indices...")
        for col in MARKET_INDICES.values():
            if col in data.columns:
                data[col] = data[col].ffill()
        
        # Add technical indicators
        logger.info("Calculating technical indicators...")
        data = self.add_technical_indicators(data, symbols)
        
        # Add lag features
        logger.info("Creating lag features...")
        data = self.add_lag_features(data, symbols)
        data = self.add_fgi_lags(data)
        data = self.add_market_cap_lags(data)
        
        # Clean data
        data = data.ffill().dropna()
        
        logger.info(f"Feature engineering complete. Final shape: {data.shape}")
        return data
