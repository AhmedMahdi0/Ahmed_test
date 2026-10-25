import time
import logging
from typing import Optional, List
import pandas as pd
import requests
import yfinance as yf

from .config import get_config, MARKET_INDICES

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataFetchError(Exception):
    """Custom exception for data fetching errors."""
    pass


class CryptoDataFetcher:
    """Fetches cryptocurrency and market data from various sources."""
    
    def __init__(self):
        self.config = get_config()
    
    def fetch_crypto_data(self, 
                          symbols: Optional[List[str]] = None,
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch cryptocurrency price and volume data.
        
        Args:
            symbols: List of crypto symbols to fetch. If None, uses enabled cryptos from config.
            start_date: Start date for data. If None, uses config value.
            end_date: End date for data. If None, uses config value.
            
        Returns:
            DataFrame with price and volume columns for each cryptocurrency.
            
        Raises:
            DataFetchError: If critical data cannot be fetched.
        """
        start_date = start_date or self.config.app_config.start_date
        end_date = end_date or self.config.app_config.end_date
        
        if symbols is None:
            crypto_tickers = self.config.get_crypto_tickers()
        else:
            crypto_tickers = {
                c.ticker: c.symbol 
                for c in self.config.cryptocurrencies 
                if c.symbol in symbols
            }
        
        crypto_data = pd.DataFrame()
        failed_downloads = []
        
        logger.info(f"Fetching cryptocurrency data from {start_date} to {end_date}")
        
        for ticker, symbol in crypto_tickers.items():
            try:
                logger.info(f"Downloading {symbol} ({ticker})...")
                data = yf.download(
                    ticker, 
                    start=start_date, 
                    end=end_date, 
                    progress=False, auto_adjust=True
                )
                time.sleep(self.config.app_config.api_delay_seconds)
                
                if not data.empty:
                    # Handle MultiIndex columns from yfinance
                    if isinstance(data.columns, pd.MultiIndex):
                        close_col = ('Close', ticker)
                        volume_col = ('Volume', ticker)
                        if close_col in data.columns:
                            crypto_data[f'{symbol}_Price'] = data[close_col]
                            crypto_data[f'{symbol}_Volume'] = data[volume_col]
                        else:
                            # Fallback for single ticker
                            crypto_data[f'{symbol}_Price'] = data['Close'].iloc[:, 0] if isinstance(data['Close'], pd.DataFrame) else data['Close']
                            crypto_data[f'{symbol}_Volume'] = data['Volume'].iloc[:, 0] if isinstance(data['Volume'], pd.DataFrame) else data['Volume']
                    else:
                        crypto_data[f'{symbol}_Price'] = data['Close']
                        crypto_data[f'{symbol}_Volume'] = data['Volume']
                    logger.info(f"Successfully downloaded {symbol}")
                else:
                    logger.warning(f"No data available for {symbol}")
                    failed_downloads.append(symbol)
                    
            except Exception as e:
                logger.error(f"Error downloading {symbol}: {e}")
                failed_downloads.append(symbol)
        
        if len(failed_downloads) == len(crypto_tickers):
            raise DataFetchError("Failed to download any cryptocurrency data")
        
        if failed_downloads:
            logger.warning(f"Failed to download: {failed_downloads}")
        
        return crypto_data.loc[:end_date]
    
    def fetch_market_indices(self,
                             start_date: Optional[str] = None,
                             end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch global market indices data.
        
        Args:
            start_date: Start date for data.
            end_date: End date for data.
            
        Returns:
            DataFrame with closing prices for each index.
        """
        start_date = start_date or self.config.app_config.start_date
        end_date = end_date or self.config.app_config.end_date
        
        indices_data = pd.DataFrame()
        
        logger.info("Fetching global market indices...")
        
        for ticker, name in MARKET_INDICES.items():
            try:
                logger.info(f"Downloading {name} ({ticker})...")
                data = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    progress=False, auto_adjust=True
                )
                time.sleep(self.config.app_config.api_delay_seconds)
                
                if not data.empty:
                    # Handle MultiIndex columns
                    if isinstance(data.columns, pd.MultiIndex):
                        close_col = ('Close', ticker)
                        if close_col in data.columns:
                            indices_data[name] = data[close_col]
                        else:
                            indices_data[name] = data['Close'].iloc[:, 0] if isinstance(data['Close'], pd.DataFrame) else data['Close']
                    else:
                        indices_data[name] = data['Close']
                    logger.info(f"Successfully downloaded {name}")
                else:
                    logger.warning(f"No data available for {name}")
                    
            except Exception as e:
                logger.error(f"Error downloading {name}: {e}")
        
        return indices_data.loc[:end_date]
    
    def fetch_fear_greed_index(self,
                               start_date: Optional[str] = None,
                               end_date: Optional[str] = None) -> pd.Series:
        """
        Fetch Fear & Greed Index data from alternative.me API.
        
        Args:
            start_date: Start date for data.
            end_date: End date for data.
            
        Returns:
            Series with FGI values indexed by date.
        """
        start_date = start_date or self.config.app_config.start_date
        end_date = end_date or self.config.app_config.end_date
        
        logger.info("Fetching Fear & Greed Index...")
        
        try:
            url = "https://api.alternative.me/fng/?limit=0"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            raw_data = response.json()['data']
            df = pd.DataFrame(raw_data)
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_numeric(df['timestamp'])
            df['date'] = pd.to_datetime(df['timestamp'], unit='s')
            
            df = df.sort_values('date')
            df = df.set_index('date')
            
            # Convert value to numeric
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            fgi_series = df['value'].rename('FGI')
            
            # Filter by date range
            date_mask = (fgi_series.index >= pd.Timestamp(start_date)) & \
                       (fgi_series.index <= pd.Timestamp(end_date))
            
            logger.info("Successfully fetched Fear & Greed Index")
            return fgi_series[date_mask]
            
        except requests.RequestException as e:
            logger.error(f"Network error fetching FGI: {e}")
            return pd.Series(name='FGI', dtype=float)
        except (KeyError, ValueError) as e:
            logger.error(f"Error parsing FGI response: {e}")
            return pd.Series(name='FGI', dtype=float)
        except Exception as e:
            logger.error(f"Unexpected error fetching FGI: {e}")
            return pd.Series(name='FGI', dtype=float)
    
    def fetch_btc_dominance(self) -> Optional[float]:
        """
        Fetch current Bitcoin dominance from CoinGecko.
        
        Returns:
            Current BTC dominance percentage, or None if fetch fails.
        """
        try:
            url = "https://api.coingecko.com/api/v3/global"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            btc_dominance = response.json()['data']['market_cap_percentage']['btc']
            logger.info(f"Current BTC dominance: {btc_dominance:.2f}%")
            return btc_dominance
            
        except requests.RequestException as e:
            logger.error(f"Network error fetching BTC dominance: {e}")
            return None
        except (KeyError, ValueError) as e:
            logger.error(f"Error parsing BTC dominance response: {e}")
            return None
    
    def fetch_all_data(self, symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Fetch all data sources and combine them.
        
        Args:
            symbols: List of crypto symbols to fetch. If None, fetches all enabled cryptos.
                     Pass a single symbol like ['BTC'] to fetch only that crypto.
        
        Returns:
            Combined DataFrame with all data sources.
        """
        if symbols:
            logger.info(f"Starting data fetch for: {symbols}")
        else:
            logger.info("Starting comprehensive data fetch for all cryptos...")
        
        # Fetch crypto data (only specified symbols if provided)
        crypto_data = self.fetch_crypto_data(symbols=symbols)
        
        # Always fetch market indices (needed for features)
        indices_data = self.fetch_market_indices()
        
        # Always fetch Fear & Greed Index (needed for features)
        fgi_data = self.fetch_fear_greed_index()
        
        # Combine all data
        logger.info("Merging all data sources...")
        combined_data = pd.concat([crypto_data, indices_data, fgi_data], axis=1)
        
        logger.info(f"Combined data shape: {combined_data.shape}")
        return combined_data
