"""
Prediction service for CryptoPredictor.
Orchestrates data fetching, feature engineering, and model predictions.
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error

from .config import get_config
from .data_fetcher import CryptoDataFetcher
from .feature_engineering import FeatureEngineer
from .models import ModelManager

logger = logging.getLogger(__name__)


class PredictionService:
    """Main service for cryptocurrency price predictions."""
    
    def __init__(self):
        self.config = get_config()
        self.data_fetcher = CryptoDataFetcher()
        self.feature_engineer = FeatureEngineer()
        self.model_manager = ModelManager()
        self.data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
    
    def fetch_and_prepare_data(self, 
                                 target_symbol: Optional[str] = None,
                                 force_refresh: bool = False) -> pd.DataFrame:
        """
        Fetch data and prepare features for a specific cryptocurrency.
        
        Args:
            target_symbol: Symbol of crypto to fetch (e.g., 'BTC'). If None, fetches all.
            force_refresh: Force re-fetching data even if cached.
            
        Returns:
            Processed DataFrame with all features.
        """
        # Check if we already have data for this symbol
        if self.processed_data is not None and not force_refresh:
            if target_symbol is None:
                logger.info("Using cached processed data (all cryptos)")
                return self.processed_data
            # Check if the target symbol exists in cached data
            price_col = f"{target_symbol}_Price"
            if price_col in self.processed_data.columns:
                logger.info(f"Using cached processed data for {target_symbol}")
                return self.processed_data
        
        logger.info(f"Fetching and preparing data for {target_symbol or 'all cryptos'}...")
        
        # Fetch raw data - only the target symbol if specified
        symbols = [target_symbol] if target_symbol else None
        self.data = self.data_fetcher.fetch_all_data(symbols=symbols)
        
        # Process features
        self.processed_data = self.feature_engineer.process_data(self.data)
        
        return self.processed_data
    
    def train_models(self, 
                     target_symbol: str,
                     model_types: Optional[List[str]] = None) -> Dict:
        """
        Train models for a specific cryptocurrency.
        
        Args:
            target_symbol: Symbol of crypto to predict (e.g., 'BTC').
            model_types: List of model types to train. If None, trains all.
            
        Returns:
            Dictionary with training results.
        """
        if self.processed_data is None:
            self.fetch_and_prepare_data()
        
        # Prepare features for target
        X, y = self.feature_engineer.prepare_features(self.processed_data, target_symbol)
        
        # Prepare data for training (pass pandas objects to preserve indexing)
        data = self.model_manager.prepare_data(
            X, 
            y,
            train_ratio=self.config.app_config.train_test_split
        )
        
        # Train models
        if model_types is None:
            self.model_manager.train_all_models(data)
        else:
            for model_type in model_types:
                self.model_manager.train_model(model_type, data)
        
        return {
            'target_symbol': target_symbol,
            'models_trained': list(self.model_manager.models.keys()),
            'train_size': data['train_size'],
            'test_size': len(X) - data['train_size']
        }
    
    def get_predictions(self, 
                        target_symbol: str,
                        include_confidence: bool = True) -> Dict:
        """
        Get predictions for a cryptocurrency.
        
        Args:
            target_symbol: Symbol to predict.
            include_confidence: Whether to include confidence intervals.
            
        Returns:
            Dictionary with predictions and metadata.
        """
        if not self.model_manager.models:
            raise ValueError("No models trained. Call train_models() first.")
        
        if self.processed_data is None:
            self.fetch_and_prepare_data()
        
        # Prepare test data (pass pandas objects so prepare_data can preserve indices)
        X, y = self.feature_engineer.prepare_features(self.processed_data, target_symbol)
        data = self.model_manager.prepare_data(X, y)
        
        # Get predictions from all models
        predictions = {}
        for model_type in self.model_manager.models:
            pred = self.model_manager.predict(model_type, data['X_test'], inverse_transform=True)
            predictions[model_type] = pred.flatten()
        
        # Get actual values
        y_actual = self.model_manager.scalers['y'].inverse_transform(
            data['y_test'].cpu().numpy()
        ).flatten()
        
        # Get dates for test period
        train_size = data['train_size']
        test_dates = self.processed_data.index[train_size:].tolist()
        
        result = {
            'target_symbol': target_symbol,
            'dates': test_dates,
            'actual': y_actual,
            'predictions': predictions
        }
        
        if include_confidence:
            result['confidence_intervals'] = self._calculate_confidence_intervals(
                y_actual, predictions
            )
        
        return result
    
    def _calculate_confidence_intervals(self, 
                                        y_actual: np.ndarray, 
                                        predictions: Dict[str, np.ndarray],
                                        confidence: float = 0.95) -> Dict:
        """Calculate confidence intervals for predictions."""
        z_score = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%
        
        intervals = {}
        for model_type, pred in predictions.items():
            errors = y_actual - pred
            std_dev = np.std(errors)
            margin = z_score * std_dev
            intervals[model_type] = {
                'lower': pred - margin,
                'upper': pred + margin,
                'std_dev': std_dev
            }
        
        return intervals
    
    def evaluate_models(self, target_symbol: str) -> pd.DataFrame:
        """
        Evaluate model performance with various metrics.
        
        Args:
            target_symbol: Symbol to evaluate.
            
        Returns:
            DataFrame with evaluation metrics.
        """
        results = self.get_predictions(target_symbol, include_confidence=False)
        y_actual = results['actual']
        
        metrics = {}
        for model_type, pred in results['predictions'].items():
            mse = mean_squared_error(y_actual, pred)
            metrics[model_type] = {
                'MSE': mse,
                'RMSE': np.sqrt(mse),
                'MAE': mean_absolute_error(y_actual, pred),
                'MAPE': np.mean(np.abs((y_actual - pred) / y_actual)) * 100
            }
        
        return pd.DataFrame(metrics).T.round(3)
    
    def get_future_predictions(self, 
                               target_symbol: str,
                               days_ahead: int = 3) -> pd.DataFrame:
        """
        Get predictions including future days using autoregressive prediction.
        
        Args:
            target_symbol: Symbol to predict.
            days_ahead: Number of days to predict ahead.
            
        Returns:
            DataFrame with predictions and confidence intervals.
        """
        if not self.model_manager.models:
            raise ValueError("Models not trained. Train models first.")
        
        if self.processed_data is None:
            self.fetch_and_prepare_data()
        
        # Get current predictions for historical context
        predictions = self.get_predictions(target_symbol)
        
        # Get the last row of features for autoregressive prediction
        X, y = self.feature_engineer.prepare_features(self.processed_data, target_symbol)

        # Ensure features are pandas-like so `.iloc` and `.columns` work
        # If prepare_features returned numpy arrays, convert them to DataFrame/Series
        if not hasattr(X, 'iloc'):
            X = pd.DataFrame(X)
        if not hasattr(y, 'iloc'):
            y = pd.Series(np.asarray(y).flatten())
        
        # Get the current/last known price from the raw data (like notebook does with y)
        price_col = f'{target_symbol}_Price'
        last_known_price = None
        if price_col in self.processed_data.columns:
            col_series = self.processed_data[price_col]
            last_known_price = float(col_series.iloc[-1] if hasattr(col_series, 'iloc') else col_series[-1])
        
        # Prepare result DataFrame
        result_data = []
        
        # Add last 7 historical days (model predictions only, no "actual")
        for i in range(-7, 0):
            idx = len(predictions['dates']) + i
            if idx >= 0:
                row_date = predictions['dates'][idx]
                row = {
                    'date': row_date,
                    'type': 'historical'
                }
                
                for model, pred in predictions['predictions'].items():
                    row[model] = float(pred[idx]) if pred[idx] is not None and not np.isnan(pred[idx]) else None
                    if 'confidence_intervals' in predictions:
                        ci = predictions['confidence_intervals'][model]
                        row[f'{model}_lower'] = float(ci['lower'][idx])
                        row[f'{model}_upper'] = float(ci['upper'][idx])
                result_data.append(row)
        
        # Add the last known actual price as a separate field in the response
        # This will be returned at the top level, not in each row
        
        # Autoregressive future predictions
        # Start with the last known features
        # Ensure last_features is a DataFrame-like object; handle numpy arrays defensively
        import pandas as pd
        if hasattr(X, 'iloc'):
            last_features = X.iloc[-1:].copy()
        else:
            # convert last row to DataFrame with generic column names if none available
            cols = X.columns if hasattr(X, 'columns') else [f'col{i}' for i in range(X.shape[1])]
            last_features = pd.DataFrame(X[-1:], columns=cols)
        last_date = predictions['dates'][-1]
        
        # Get price lag column names for the target
        price_lag_cols = [f'{target_symbol}_Price_lag{i}' for i in range(1, 8)]
        price_lag_cols = [col for col in price_lag_cols if col in X.columns]
        
        # Store predictions from each model to use for rolling
        model_predictions = {model: [] for model in self.model_manager.models}
        
        # Get the scaler for transforming features
        scaler_X = self.model_manager.scalers.get('X')
        scaler_y = self.model_manager.scalers.get('y')
        
        for day in range(days_ahead):
            future_date = last_date + timedelta(days=day + 1)
            row = {
                'date': future_date,
                'type': 'forecast'
            }
            
            # Scale the current features
            features_scaled = scaler_X.transform(last_features.values)
            features_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(self.model_manager.device)
            
            # Get predictions from all models
            day_predictions = []
            for model_type, model in self.model_manager.models.items():
                model.eval()
                with torch.no_grad():
                    pred_scaled = model(features_tensor).cpu().numpy()
                    pred = scaler_y.inverse_transform(pred_scaled).flatten()[0]
                    row[model_type] = pred
                    model_predictions[model_type].append(pred)
                    day_predictions.append(pred)
            
            # Calculate ensemble prediction (average of all models)
            ensemble_pred = np.mean(day_predictions)
            
            # Add confidence intervals based on model variance
            std_dev = np.std(day_predictions)
            for model_type in self.model_manager.models:
                row[f'{model_type}_lower'] = row[model_type] - 1.96 * std_dev
                row[f'{model_type}_upper'] = row[model_type] + 1.96 * std_dev
            
            result_data.append(row)
            
            # Update features for next day prediction (autoregressive step)
            # Shift price lags: lag1 <- predicted, lag2 <- lag1, etc.
            if price_lag_cols:
                new_features = last_features.copy()
                
                # Shift existing lags
                for i in range(len(price_lag_cols) - 1, 0, -1):
                    col_current = f'{target_symbol}_Price_lag{i}'
                    col_prev = f'{target_symbol}_Price_lag{i+1}'
                    if col_current in new_features.columns and col_prev in new_features.columns:
                        new_features[col_prev] = new_features[col_current].values
                
                # Set lag1 to the ensemble predicted price
                lag1_col = f'{target_symbol}_Price_lag1'
                if lag1_col in new_features.columns:
                    new_features[lag1_col] = ensemble_pred
                
                last_features = new_features
        
        # Return DataFrame and current price
        return pd.DataFrame(result_data), last_known_price
    
    def classify_price_change(self, 
                              true_price: float, 
                              predicted_price: float) -> int:
        """
        Classify predicted price change into categories.
        
        Args:
            true_price: True price at time t.
            predicted_price: Predicted price at time t+1.
            
        Returns:
            Category from -7 to 7.
        """
        if true_price == 0:
            return -7 if predicted_price == 0 else 0
        
        pct_change = ((predicted_price - true_price) / true_price) * 100
        
        if -2 < pct_change < 2: return 0
        elif 2 <= pct_change < 5: return 1
        elif 5 <= pct_change < 10: return 2
        elif 10 <= pct_change < 20: return 3
        elif 20 <= pct_change < 40: return 4
        elif 40 <= pct_change < 60: return 5
        elif 60 <= pct_change < 100: return 6
        elif pct_change >= 100: return 7
        elif -5 < pct_change <= -2: return -1
        elif -10 < pct_change <= -5: return -2
        elif -20 < pct_change <= -10: return -3
        elif -40 < pct_change <= -20: return -4
        elif -60 < pct_change <= -40: return -5
        elif -100 < pct_change <= -60: return -6
        else: return -7
    
    def get_category_names(self) -> Dict[int, str]:
        """Get mapping of category numbers to names."""
        return {
            -7: "<= -100%",
            -6: "-100% to -60%",
            -5: "-60% to -40%",
            -4: "-40% to -20%",
            -3: "-20% to -10%",
            -2: "-10% to -5%",
            -1: "-5% to -2%",
            0: "-2% to 2%",
            1: "2% to 5%",
            2: "5% to 10%",
            3: "10% to 20%",
            4: "20% to 40%",
            5: "40% to 60%",
            6: "60% to 100%",
            7: ">=100%"
        }
    
    def save_results(self, 
                     target_symbol: str,
                     filename: Optional[str] = None) -> str:
        """
        Save prediction results to CSV.
        
        Args:
            target_symbol: Symbol predicted.
            filename: Output filename. Auto-generated if None.
            
        Returns:
            Path to saved file.
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{target_symbol}_predictions_{timestamp}.csv"
        
        predictions_df = self.get_future_predictions(target_symbol)
        output_path = self.config.get_output_path(filename)
        predictions_df.to_csv(output_path, index=False)
        
        logger.info(f"Results saved to {output_path}")
        return output_path
    
    def save_models(self) -> str:
        """Save trained models to disk."""
        models_dir = self.config.get_output_path("models")
        self.model_manager.save_models(models_dir)
        return models_dir
    
    def load_models(self, input_dim: int = None) -> None:
        """Load trained models from disk."""
        models_dir = self.config.get_output_path("models")
        self.model_manager.load_models(models_dir, input_dim)
    
    def get_available_cryptos(self) -> List[Dict]:
        """Get list of available cryptocurrencies."""
        return [
            {
                'symbol': c.symbol,
                'name': c.name,
                'ticker': c.ticker,
                'enabled': c.enabled
            }
            for c in self.config.cryptocurrencies
        ]
    
    def get_btc_dominance(self) -> Optional[float]:
        """Get current BTC dominance."""
        return self.data_fetcher.fetch_btc_dominance()
