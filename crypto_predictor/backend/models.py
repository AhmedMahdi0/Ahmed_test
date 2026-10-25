"""
ML Models module for CryptoPredictor.
Contains GRU, LSTM, and Hybrid Transformer models.
"""

import logging
from typing import Optional, Dict, Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

from .config import get_config

logger = logging.getLogger(__name__)

class GRUModel(nn.Module):
    """Bidirectional GRU model for time series prediction."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim * 2, 1)
    
    def forward(self, x):
        x, _ = self.gru(x.unsqueeze(1))
        return self.fc(x[:, -1, :])
        


class LSTMModel(nn.Module):
    """Bidirectional LSTM model for time series prediction."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim * 2, 1)
    
    def forward(self, x):
        x, _ = self.lstm(x.unsqueeze(1))
        return self.fc(x[:, -1, :])


class HybridTransformerLSTM(nn.Module):
    """Hybrid Transformer + LSTM model."""
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int = 128, 
                 num_heads: int = 8, 
                 num_layers: int = 4, 
                 lstm_hidden_dim: int = 64, 
                 lstm_layers: int = 2, 
                 dropout: float = 0.2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # LSTM Decoder
        self.lstm = nn.LSTM(hidden_dim, lstm_hidden_dim, lstm_layers, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_dim, 1)
    
    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        x = self.transformer(x)
        x, _ = self.lstm(x)
        return self.fc(x[:, -1, :])


class HybridTransformerGRU(nn.Module):
    """Hybrid Transformer + GRU model."""
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int = 128, 
                 num_heads: int = 8, 
                 num_layers: int = 4, 
                 gru_hidden_dim: int = 64, 
                 gru_layers: int = 2, 
                 dropout: float = 0.2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # GRU Decoder
        self.gru = nn.GRU(hidden_dim, gru_hidden_dim, gru_layers, batch_first=True)
        self.fc = nn.Linear(gru_hidden_dim, 1)
    
    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        x = self.transformer(x)
        x, _ = self.gru(x)
        return self.fc(x[:, -1, :])


# ============================================
# Model Manager
# ============================================

class ModelManager:
    """Manages training, prediction, and persistence of models."""
    
    MODEL_CLASSES = {
        'gru': GRUModel,
        'lstm': LSTMModel,
        'hybrid_lstm': HybridTransformerLSTM,
        'hybrid_gru': HybridTransformerGRU
    }
    
    def __init__(self):
        self.config = get_config()
        self.models: Dict[str, nn.Module] = {}
        self.scalers: Dict[str, MinMaxScaler] = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
    
    def prepare_data(self, X, y, train_ratio: float = None) -> Dict:
        """
        Prepare and scale data for training using an 80/20 split (by default
        or per config). Uses pandas `.iloc` when available to preserve indices.

        Returns tensors (moved to device), scalers, a combined DataLoader,
        and train_size.
        """
        # Determine train ratio: use provided or config default (80%)
        if train_ratio is None:
            train_ratio = getattr(self.config.app_config, 'train_test_split', 0.8)

        # Prefer pandas slicing when possible to preserve `.iloc` semantics
        try:
            import pandas as pd
        except Exception:
            pd = None

        if pd is not None and hasattr(X, 'iloc'):
            length = len(X)
            train_size = int(train_ratio * length)
            X_train = X.iloc[:train_size]
            X_test = X.iloc[train_size:]
            y_train = y.iloc[:train_size]
            y_test = y.iloc[train_size:]
            # Convert to numpy for scalers
            X_train_arr = X_train.values
            X_test_arr = X_test.values
            y_train_arr = y_train.values
            y_test_arr = y_test.values
        else:
            X_arr = np.asarray(X)
            y_arr = np.asarray(y)
            length = X_arr.shape[0]
            train_size = int(train_ratio * length)
            X_train_arr = X_arr[:train_size]
            X_test_arr = X_arr[train_size:]
            y_train_arr = y_arr[:train_size]
            y_test_arr = y_arr[train_size:]

        # Scale data (Min-Max)
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()

        X_train_scaled = scaler_X.fit_transform(X_train_arr)
        X_test_scaled = scaler_X.transform(X_test_arr)
        y_train_scaled = scaler_y.fit_transform(y_train_arr if y_train_arr.ndim > 1 else y_train_arr.reshape(-1, 1))
        y_test_scaled = scaler_y.transform(y_test_arr if y_test_arr.ndim > 1 else y_test_arr.reshape(-1, 1))

        # Convert to tensors (move to device)
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).to(self.device)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(self.device)
        y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32).to(self.device)

        # Combined tensors for DataLoader
        combined_X = np.concatenate((X_train_scaled, X_test_scaled), axis=0)
        combined_y = np.concatenate((y_train_scaled, y_test_scaled), axis=0)
        X_tensor = torch.tensor(combined_X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(combined_y, dtype=torch.float32).view(-1, 1).to(self.device)

        # DataLoader (use config batch_size if available)
        batch_size = getattr(self.config.app_config, 'batch_size', 64)
        seed = getattr(self.config.app_config, 'seed', 42)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                generator=torch.Generator().manual_seed(int(seed)))

        data = {
            'X_train': X_train_tensor,
            'y_train': y_train_tensor,
            'X_test': X_test_tensor,
            'y_test': y_test_tensor,
            'X_tensor': X_tensor,
            'y_tensor': y_tensor,
            'scaler_X': scaler_X,
            'scaler_y': scaler_y,
            'dataloader': dataloader,
            'train_size': train_size
        }

        self.scalers['X'] = scaler_X
        self.scalers['y'] = scaler_y

        return data
    
    def train_model(self, 
                    model_type: str, 
                    data: Dict,
                    epochs: Optional[int] = None,
                    learning_rate: Optional[float] = None,
                    verbose: bool = True) -> nn.Module:
        """
        Train a specific model.
        
        Args:
            model_type: Type of model ('gru', 'lstm', 'hybrid_lstm', 'hybrid_gru').
            data: Prepared data dictionary from prepare_data().
            epochs: Number of training epochs.
            learning_rate: Learning rate.
            verbose: Whether to print progress.
            
        Returns:
            Trained model.
        """
        if model_type not in self.MODEL_CLASSES:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(self.MODEL_CLASSES.keys())}")

        input_dim = data['X_train'].shape[1]
        hidden_dim = getattr(self.config.app_config, 'hidden_dim', 64)
        num_layers = getattr(self.config.app_config, 'num_layers', 2)

        # Set default hyperparameters based on model type
        if epochs is None:
            if model_type in ['gru', 'lstm']:
                epochs = getattr(self.config.app_config, 'gru_epochs', 100)
            else:
                epochs = getattr(self.config.app_config, 'transformer_epochs', 200)

        if learning_rate is None:
            if model_type in ['gru', 'lstm']:
                learning_rate = getattr(self.config.app_config, 'learning_rate', 0.001)
            else:
                learning_rate = getattr(self.config.app_config, 'transformer_learning_rate', 0.0005)

        # Initialize model
        if model_type == 'gru':
            model = GRUModel(input_dim, hidden_dim=hidden_dim, num_layers=num_layers).to(self.device)
        elif model_type == 'lstm':
            model = LSTMModel(input_dim, hidden_dim=hidden_dim, num_layers=num_layers).to(self.device)
        elif model_type == 'hybrid_lstm':
            model = HybridTransformerLSTM(
                input_dim,
                hidden_dim=max(64, hidden_dim),
                lstm_hidden_dim=hidden_dim,
                lstm_layers=num_layers
            ).to(self.device)
        elif model_type == 'hybrid_gru':
            model = HybridTransformerGRU(
                input_dim,
                hidden_dim=max(64, hidden_dim),
                gru_hidden_dim=hidden_dim,
                gru_layers=num_layers
            ).to(self.device)
        else:
            model = self.MODEL_CLASSES[model_type](input_dim).to(self.device)

        # Optimizer and loss
        if model_type in ['hybrid_lstm', 'hybrid_gru']:
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        else:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        criterion = nn.MSELoss()

        # Training loop (GRU/LSTM on raw tensors, transformer models using DataLoader)
        logger.info(f"Training {model_type.upper()} model for {epochs} epochs...")

        if model_type in ['hybrid_lstm', 'hybrid_gru']:
            train_loader = data.get('dataloader')
            if train_loader is None:
                train_dataset = TensorDataset(data['X_train'], data['y_train'])
                train_loader = DataLoader(train_dataset, batch_size=getattr(self.config.app_config, 'batch_size', 64), shuffle=True)

            for epoch in range(epochs):
                model.train()
                total_loss = 0.0
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / max(1, len(train_loader)):.6f}")
        else:
            X_train_tensor = data['X_train']
            y_train_tensor = data['y_train']

            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                y_pred = model(X_train_tensor)
                loss = criterion(y_pred, y_train_tensor)
                loss.backward()
                optimizer.step()

                if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
        
        model.eval()
        self.models[model_type] = model
        logger.info(f"{model_type.upper()} model training complete")
        
        return model
    
    def train_all_models(self, data: Dict, verbose: bool = True) -> Dict[str, nn.Module]:
        """
        Train all available models.
        
        Args:
            data: Prepared data dictionary.
            verbose: Whether to print progress.
            
        Returns:
            Dictionary of trained models.
        """
        for model_type in self.MODEL_CLASSES.keys():
            self.train_model(model_type, data, verbose=verbose)
        
        return self.models
    
    def predict(self, 
                model_type: str, 
                X: torch.Tensor,
                inverse_transform: bool = True) -> np.ndarray:
        """
        Make predictions using a trained model.
        
        Args:
            model_type: Type of model.
            X: Input features (scaled).
            inverse_transform: Whether to inverse transform predictions.
            
        Returns:
            Predictions array.
        """
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not trained. Train it first.")
        
        model = self.models[model_type]
        model.eval()

        with torch.no_grad():
            # use detach().cpu().numpy() to match the expected output format
            predictions = model(X).detach().cpu().numpy()

        if inverse_transform and 'y' in self.scalers:
            predictions = self.scalers['y'].inverse_transform(predictions)

        return predictions
    
    def predict_all(self, X: torch.Tensor, inverse_transform: bool = True) -> Dict[str, np.ndarray]:
        """
        Make predictions using all trained models.
        
        Args:
            X: Input features.
            inverse_transform: Whether to inverse transform predictions.
            
        Returns:
            Dictionary of predictions by model type.
        """
        # Map internal model keys to human-friendly display names
        display_names = {
            'gru': 'Bi-GRU',
            'lstm': 'Bi-LSTM',
            'hybrid_lstm': 'Hybrid Transformer + LSTM',
            'hybrid_gru': 'Hybrid Transformer + GRU'
        }

        predictions = {}
        for model_type in self.models.keys():
            name = display_names.get(model_type, model_type)
            predictions[name] = self.predict(model_type, X, inverse_transform)

        return predictions
    
    def save_models(self, directory: Optional[str] = None) -> None:
        """
        Save all trained models to disk.
        
        Args:
            directory: Directory to save models. Uses config output_dir if None.
        """
        if directory is None:
            directory = self.config.get_output_path("models")
        
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        for model_type, model in self.models.items():
            path = Path(directory) / f"{model_type}_model.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_class': model.__class__.__name__,
                'input_dim': model.fc.in_features if hasattr(model, 'fc') else None
            }, path)
            logger.info(f"Saved {model_type} model to {path}")
        
        # Save scalers
        import pickle
        scalers_path = Path(directory) / "scalers.pkl"
        with open(scalers_path, 'wb') as f:
            pickle.dump(self.scalers, f)
        logger.info(f"Saved scalers to {scalers_path}")
    
    def load_models(self, directory: Optional[str] = None, input_dim: int = None) -> None:
        """
        Load models from disk.
        
        Args:
            directory: Directory containing saved models.
            input_dim: Input dimension for model initialization.
        """
        if directory is None:
            directory = self.config.get_output_path("models")
        
        directory = Path(directory)
        
        for model_type, model_class in self.MODEL_CLASSES.items():
            path = directory / f"{model_type}_model.pt"
            if path.exists():
                checkpoint = torch.load(path, map_location=self.device)
                dim = input_dim or checkpoint.get('input_dim', 41)
                model = model_class(dim).to(self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                self.models[model_type] = model
                logger.info(f"Loaded {model_type} model from {path}")
        
        # Load scalers
        import pickle
        scalers_path = directory / "scalers.pkl"
        if scalers_path.exists():
            with open(scalers_path, 'rb') as f:
                self.scalers = pickle.load(f)
            logger.info(f"Loaded scalers from {scalers_path}")
