"""
LSTM Price Predictor for Trading Bot.

This module implements an advanced LSTM-based price prediction system that:
- Predicts next 5-minute price movements using 60-period lookback
- Uses multi-feature input (OHLCV + technical indicators)
- Implements proper data preprocessing and normalization
- Includes model training, validation, and inference methods
- Saves/loads trained models with versioning
- Provides real-time inference pipeline
"""

import logging
import os
import json
import pickle
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logger = logging.getLogger(__name__)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")


@dataclass
class ModelConfig:
    """Configuration for LSTM model."""
    sequence_length: int = 60
    n_features: int = 15
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 100
    early_stopping_patience: int = 10
    validation_split: float = 0.2
    test_split: float = 0.1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelConfig':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class TrainingMetrics:
    """Training metrics tracking."""
    epoch: int
    train_loss: float
    val_loss: float
    train_accuracy: float
    val_accuracy: float
    learning_rate: float
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class ModelPerformance:
    """Model performance metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: List[List[int]]
    class_distribution: Dict[str, int]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class PriceDataset(Dataset):
    """Custom dataset for price prediction."""
    
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        """
        Initialize dataset.
        
        Args:
            sequences: Input sequences (n_samples, sequence_length, n_features)
            targets: Target labels (n_samples,)
        """
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.LongTensor(targets)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.targets[idx]


class LSTMModel(nn.Module):
    """LSTM model for price prediction."""
    
    def __init__(self, config: ModelConfig):
        """
        Initialize LSTM model.
        
        Args:
            config: Model configuration
        """
        super(LSTMModel, self).__init__()
        self.config = config
        
        # Input batch normalization
        self.input_bn = nn.BatchNorm1d(config.n_features)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=config.n_features,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(config.dropout)
        
        # Batch normalization for LSTM output
        self.lstm_bn = nn.BatchNorm1d(config.hidden_size)
        
        # Fully connected layers
        self.fc1 = nn.Linear(config.hidden_size, config.hidden_size // 2)
        self.fc1_bn = nn.BatchNorm1d(config.hidden_size // 2)
        self.fc2 = nn.Linear(config.hidden_size // 2, config.hidden_size // 4)
        self.fc2_bn = nn.BatchNorm1d(config.hidden_size // 4)
        
        # Output layer (3 classes: down, neutral, up)
        self.output = nn.Linear(config.hidden_size // 4, 3)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, sequence_length, n_features)
            
        Returns:
            Output probabilities (batch_size, 3)
        """
        batch_size, seq_len, n_features = x.size()
        
        # Apply input batch normalization
        x = x.view(-1, n_features)
        x = self.input_bn(x)
        x = x.view(batch_size, seq_len, n_features)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last output
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Apply batch normalization and dropout
        x = self.lstm_bn(last_output)
        x = self.dropout(x)
        
        # Fully connected layers
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = self.dropout(x)
        
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = self.dropout(x)
        
        # Output layer
        x = self.output(x)
        
        return F.softmax(x, dim=1)


class DataPreprocessor:
    """Data preprocessing pipeline for LSTM model."""
    
    def __init__(self, sequence_length: int = 60):
        """
        Initialize preprocessor.
        
        Args:
            sequence_length: Length of input sequences
        """
        self.sequence_length = sequence_length
        self.feature_scaler = RobustScaler()
        self.price_scaler = StandardScaler()
        self.is_fitted = False
        
        # Feature names for reference
        self.feature_names = [
            'open', 'high', 'low', 'close', 'volume',
            'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower',
            'sma_20', 'ema_12', 'ema_26', 'atr', 'volume_sma'
        ]
    
    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract features from price dataframe.
        
        Args:
            df: Price dataframe with OHLCV and indicators
            
        Returns:
            Feature array (n_samples, n_features)
        """
        try:
            features = []
            
            # Basic OHLCV features
            features.extend([
                df['open'].values,
                df['high'].values,
                df['low'].values,
                df['close'].values,
                df['volume'].values
            ])
            
            # Technical indicators
            features.extend([
                df.get('rsi', pd.Series(50, index=df.index)).values,
                df.get('macd', pd.Series(0, index=df.index)).values,
                df.get('macd_signal', pd.Series(0, index=df.index)).values,
                df.get('bb_upper', df['close']).values,
                df.get('bb_lower', df['close']).values,
                df.get('sma_20', df['close']).values,
                df.get('ema_12', df['close']).values,
                df.get('ema_26', df['close']).values,
                df.get('atr', pd.Series(1, index=df.index)).values,
                df.get('volume_sma', df['volume']).values
            ])
            
            # Stack features
            feature_array = np.column_stack(features)
            
            # Handle NaN values
            feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=0.0, neginf=0.0)
            
            return feature_array
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            # Return default features
            n_samples = len(df)
            return np.zeros((n_samples, len(self.feature_names)))
    
    def create_labels(self, prices: np.ndarray, threshold: float = 0.001) -> np.ndarray:
        """
        Create labels for price direction prediction.
        
        Args:
            prices: Close prices
            threshold: Minimum price change threshold
            
        Returns:
            Labels array (0: down, 1: neutral, 2: up)
        """
        try:
            # Calculate price changes
            price_changes = np.diff(prices) / prices[:-1]
            
            # Create labels
            labels = np.ones(len(price_changes), dtype=int)  # Default: neutral
            labels[price_changes < -threshold] = 0  # Down
            labels[price_changes > threshold] = 2   # Up
            
            return labels
            
        except Exception as e:
            logger.error(f"Error creating labels: {e}")
            return np.ones(len(prices) - 1, dtype=int)
    
    def create_sequences(self, features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training.
        
        Args:
            features: Feature array
            labels: Labels array
            
        Returns:
            Tuple of (sequences, targets)
        """
        try:
            sequences = []
            targets = []
            
            for i in range(self.sequence_length, len(features)):
                # Get sequence
                seq = features[i - self.sequence_length:i]
                sequences.append(seq)
                
                # Get target (if available)
                if i - 1 < len(labels):
                    targets.append(labels[i - 1])
            
            return np.array(sequences), np.array(targets)
            
        except Exception as e:
            logger.error(f"Error creating sequences: {e}")
            return np.array([]), np.array([])
    
    def fit_transform(self, df: pd.DataFrame, price_threshold: float = 0.001) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit preprocessor and transform data.
        
        Args:
            df: Price dataframe
            price_threshold: Price change threshold for labels
            
        Returns:
            Tuple of (sequences, targets)
        """
        try:
            # Extract features
            features = self.extract_features(df)
            
            # Fit and transform features
            features_scaled = self.feature_scaler.fit_transform(features)
            
            # Create labels
            labels = self.create_labels(df['close'].values, price_threshold)
            
            # Create sequences
            sequences, targets = self.create_sequences(features_scaled, labels)
            
            self.is_fitted = True
            
            logger.info(f"Preprocessed data: {len(sequences)} sequences, {len(targets)} targets")
            return sequences, targets
            
        except Exception as e:
            logger.error(f"Error in fit_transform: {e}")
            return np.array([]), np.array([])
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using fitted preprocessor.
        
        Args:
            df: Price dataframe
            
        Returns:
            Scaled feature sequences
        """
        try:
            if not self.is_fitted:
                raise ValueError("Preprocessor not fitted. Call fit_transform first.")
            
            # Extract features
            features = self.extract_features(df)
            
            # Transform features
            features_scaled = self.feature_scaler.transform(features)
            
            # Create sequences (without labels)
            if len(features_scaled) >= self.sequence_length:
                sequences = []
                for i in range(self.sequence_length, len(features_scaled) + 1):
                    seq = features_scaled[i - self.sequence_length:i]
                    sequences.append(seq)
                
                return np.array(sequences)
            else:
                logger.warning(f"Insufficient data for sequence creation: {len(features_scaled)} < {self.sequence_length}")
                return np.array([])
            
        except Exception as e:
            logger.error(f"Error in transform: {e}")
            return np.array([])


class LSTMPricePredictor:
    """
    LSTM-based price predictor with comprehensive training and inference pipeline.
    """
    
    def __init__(self, 
                 model_dir: str = "models",
                 config: Optional[ModelConfig] = None):
        """
        Initialize LSTM price predictor.
        
        Args:
            model_dir: Directory to save/load models
            config: Model configuration
        """
        self.model_dir = model_dir
        self.config = config or ModelConfig()
        
        # Initialize components
        self.model = None
        self.preprocessor = DataPreprocessor(self.config.sequence_length)
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.training_history: List[TrainingMetrics] = []
        self.model_performance: Optional[ModelPerformance] = None
        
        # Model versioning
        self.model_version = "1.0.0"
        self.model_metadata = {}
        
        # Ensure model directory exists
        os.makedirs(model_dir, exist_ok=True)
        
        logger.info(f"LSTMPricePredictor initialized with config: {self.config}")
    
    def _create_model(self) -> LSTMModel:
        """Create LSTM model."""
        model = LSTMModel(self.config).to(device)
        return model
    
    def _split_data(self, sequences: np.ndarray, targets: np.ndarray) -> Tuple[
        Tuple[np.ndarray, np.ndarray],  # train
        Tuple[np.ndarray, np.ndarray],  # val
        Tuple[np.ndarray, np.ndarray]   # test
    ]:
        """Split data into train, validation, and test sets."""
        try:
            n_samples = len(sequences)
            
            # Calculate split indices
            test_size = int(n_samples * self.config.test_split)
            val_size = int(n_samples * self.config.validation_split)
            train_size = n_samples - test_size - val_size
            
            # Split data chronologically (important for time series)
            train_seq = sequences[:train_size]
            train_targets = targets[:train_size]
            
            val_seq = sequences[train_size:train_size + val_size]
            val_targets = targets[train_size:train_size + val_size]
            
            test_seq = sequences[train_size + val_size:]
            test_targets = targets[train_size + val_size:]
            
            logger.info(f"Data split - Train: {len(train_seq)}, Val: {len(val_seq)}, Test: {len(test_seq)}")
            
            return (train_seq, train_targets), (val_seq, val_targets), (test_seq, test_targets)
            
        except Exception as e:
            logger.error(f"Error splitting data: {e}")
            return (sequences, targets), (np.array([]), np.array([])), (np.array([]), np.array([]))   
 def train(self, df: pd.DataFrame, price_threshold: float = 0.001) -> bool:
        """
        Train the LSTM model.
        
        Args:
            df: Training dataframe with OHLCV and indicators
            price_threshold: Price change threshold for labels
            
        Returns:
            True if training successful, False otherwise
        """
        try:
            logger.info("Starting LSTM model training...")
            
            # Preprocess data
            sequences, targets = self.preprocessor.fit_transform(df, price_threshold)
            
            if len(sequences) == 0:
                logger.error("No sequences created from data")
                return False
            
            # Check class distribution
            unique, counts = np.unique(targets, return_counts=True)
            class_dist = dict(zip(unique, counts))
            logger.info(f"Class distribution: {class_dist}")
            
            # Split data
            (train_seq, train_targets), (val_seq, val_targets), (test_seq, test_targets) = self._split_data(sequences, targets)
            
            if len(train_seq) == 0:
                logger.error("No training data available")
                return False
            
            # Create data loaders
            train_dataset = PriceDataset(train_seq, train_targets)
            train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
            
            val_dataset = PriceDataset(val_seq, val_targets) if len(val_seq) > 0 else None
            val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False) if val_dataset else None
            
            # Create model
            self.model = self._create_model()
            
            # Setup optimizer and scheduler
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate, weight_decay=1e-5)
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5, verbose=True)
            
            # Training loop
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(self.config.num_epochs):
                # Training phase
                train_loss, train_acc = self._train_epoch(train_loader)
                
                # Validation phase
                val_loss, val_acc = self._validate_epoch(val_loader) if val_loader else (0.0, 0.0)
                
                # Learning rate scheduling
                if val_loader:
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step(train_loss)
                
                # Get current learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Log metrics
                metrics = TrainingMetrics(
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    train_accuracy=train_acc,
                    val_accuracy=val_acc,
                    learning_rate=current_lr,
                    timestamp=datetime.now()
                )
                self.training_history.append(metrics)
                
                # Early stopping
                if val_loader and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    self._save_checkpoint('best_model.pth')
                else:
                    patience_counter += 1
                
                # Log progress
                if epoch % 10 == 0 or epoch == self.config.num_epochs - 1:
                    logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                               f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, LR: {current_lr:.6f}")
                
                # Early stopping check
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            # Load best model
            if os.path.exists(os.path.join(self.model_dir, 'best_model.pth')):
                self._load_checkpoint('best_model.pth')
            
            # Evaluate on test set
            if len(test_seq) > 0:
                test_dataset = PriceDataset(test_seq, test_targets)
                test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)
                self.model_performance = self._evaluate_model(test_loader)
                logger.info(f"Test Performance: {self.model_performance}")
            
            # Save final model
            self.save_model()
            
            logger.info("Training completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            return False
    
    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for batch_idx, (sequences, targets) in enumerate(train_loader):
            sequences, targets = sequences.to(device), targets.to(device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(sequences)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += targets.size(0)
            correct_predictions += (predicted == targets).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy
    
    def _validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch."""
        if not val_loader:
            return 0.0, 0.0
        
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences, targets = sequences.to(device), targets.to(device)
                
                # Forward pass
                outputs = self.model(sequences)
                loss = self.criterion(outputs, targets)
                
                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_predictions += targets.size(0)
                correct_predictions += (predicted == targets).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy
    
    def _evaluate_model(self, test_loader: DataLoader) -> ModelPerformance:
        """Evaluate model performance on test set."""
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for sequences, targets in test_loader:
                sequences, targets = sequences.to(device), targets.to(device)
                
                outputs = self.model(sequences)
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        precision = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_targets, all_predictions, average='weighted', zero_division=0)
        f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        
        # Class distribution
        unique, counts = np.unique(all_targets, return_counts=True)
        class_dist = {f"class_{i}": int(count) for i, count in zip(unique, counts)}
        
        return ModelPerformance(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            confusion_matrix=cm.tolist(),
            class_distribution=class_dist,
            timestamp=datetime.now()
        )
    
    def predict(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Make prediction on new data.
        
        Args:
            df: Price dataframe with OHLCV and indicators
            
        Returns:
            Prediction dictionary with probabilities and class
        """
        try:
            if self.model is None:
                logger.error("Model not trained or loaded")
                return None
            
            if not self.preprocessor.is_fitted:
                logger.error("Preprocessor not fitted")
                return None
            
            # Preprocess data
            sequences = self.preprocessor.transform(df)
            
            if len(sequences) == 0:
                logger.warning("No sequences created for prediction")
                return None
            
            # Use the last sequence for prediction
            last_sequence = sequences[-1:]  # Keep batch dimension
            
            self.model.eval()
            with torch.no_grad():
                sequence_tensor = torch.FloatTensor(last_sequence).to(device)
                output = self.model(sequence_tensor)
                probabilities = output.cpu().numpy()[0]  # Remove batch dimension
                predicted_class = np.argmax(probabilities)
            
            # Map class to direction
            class_mapping = {0: 'down', 1: 'neutral', 2: 'up'}
            
            return {
                'predicted_class': predicted_class,
                'predicted_direction': class_mapping[predicted_class],
                'probabilities': {
                    'down': float(probabilities[0]),
                    'neutral': float(probabilities[1]),
                    'up': float(probabilities[2])
                },
                'confidence': float(np.max(probabilities)),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return None
    
    def predict_batch(self, df: pd.DataFrame) -> Optional[List[Dict[str, Any]]]:
        """
        Make predictions on multiple sequences.
        
        Args:
            df: Price dataframe with OHLCV and indicators
            
        Returns:
            List of prediction dictionaries
        """
        try:
            if self.model is None:
                logger.error("Model not trained or loaded")
                return None
            
            # Preprocess data
            sequences = self.preprocessor.transform(df)
            
            if len(sequences) == 0:
                logger.warning("No sequences created for prediction")
                return None
            
            self.model.eval()
            predictions = []
            
            with torch.no_grad():
                # Process in batches
                batch_size = self.config.batch_size
                for i in range(0, len(sequences), batch_size):
                    batch_sequences = sequences[i:i + batch_size]
                    sequence_tensor = torch.FloatTensor(batch_sequences).to(device)
                    
                    outputs = self.model(sequence_tensor)
                    probabilities = outputs.cpu().numpy()
                    
                    # Process each prediction in batch
                    for j, probs in enumerate(probabilities):
                        predicted_class = np.argmax(probs)
                        class_mapping = {0: 'down', 1: 'neutral', 2: 'up'}
                        
                        predictions.append({
                            'sequence_index': i + j,
                            'predicted_class': predicted_class,
                            'predicted_direction': class_mapping[predicted_class],
                            'probabilities': {
                                'down': float(probs[0]),
                                'neutral': float(probs[1]),
                                'up': float(probs[2])
                            },
                            'confidence': float(np.max(probs))
                        })
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making batch predictions: {e}")
            return None
    
    def save_model(self, version: Optional[str] = None) -> bool:
        """
        Save trained model with versioning.
        
        Args:
            version: Model version (optional)
            
        Returns:
            True if save successful, False otherwise
        """
        try:
            if self.model is None:
                logger.error("No model to save")
                return False
            
            # Set version
            if version:
                self.model_version = version
            
            # Create model metadata
            self.model_metadata = {
                'version': self.model_version,
                'config': self.config.to_dict(),
                'training_history': [metrics.to_dict() for metrics in self.training_history],
                'model_performance': self.model_performance.to_dict() if self.model_performance else None,
                'feature_names': self.preprocessor.feature_names,
                'created_at': datetime.now().isoformat(),
                'pytorch_version': torch.__version__
            }
            
            # Save model state
            model_path = os.path.join(self.model_dir, f'lstm_model_v{self.model_version}.pth')
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
                'config': self.config.to_dict(),
                'metadata': self.model_metadata
            }, model_path)
            
            # Save preprocessor
            preprocessor_path = os.path.join(self.model_dir, f'preprocessor_v{self.model_version}.pkl')
            with open(preprocessor_path, 'wb') as f:
                pickle.dump(self.preprocessor, f)
            
            # Save metadata
            metadata_path = os.path.join(self.model_dir, f'metadata_v{self.model_version}.json')
            with open(metadata_path, 'w') as f:
                json.dump(self.model_metadata, f, indent=2)
            
            logger.info(f"Model saved successfully: version {self.model_version}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, version: Optional[str] = None) -> bool:
        """
        Load trained model.
        
        Args:
            version: Model version to load (latest if None)
            
        Returns:
            True if load successful, False otherwise
        """
        try:
            # Find model version to load
            if version is None:
                # Find latest version
                model_files = [f for f in os.listdir(self.model_dir) if f.startswith('lstm_model_v') and f.endswith('.pth')]
                if not model_files:
                    logger.error("No saved models found")
                    return False
                
                # Sort by version (simple string sort should work for semantic versioning)
                model_files.sort(reverse=True)
                latest_file = model_files[0]
                version = latest_file.replace('lstm_model_v', '').replace('.pth', '')
            
            self.model_version = version
            
            # Load model
            model_path = os.path.join(self.model_dir, f'lstm_model_v{version}.pth')
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return False
            
            checkpoint = torch.load(model_path, map_location=device)
            
            # Load config
            self.config = ModelConfig.from_dict(checkpoint['config'])
            
            # Create and load model
            self.model = self._create_model()
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load metadata
            self.model_metadata = checkpoint.get('metadata', {})
            
            # Load preprocessor
            preprocessor_path = os.path.join(self.model_dir, f'preprocessor_v{version}.pkl')
            if os.path.exists(preprocessor_path):
                with open(preprocessor_path, 'rb') as f:
                    self.preprocessor = pickle.load(f)
            else:
                logger.warning("Preprocessor file not found, creating new one")
                self.preprocessor = DataPreprocessor(self.config.sequence_length)
            
            # Load training history if available
            if 'training_history' in self.model_metadata:
                self.training_history = [
                    TrainingMetrics(
                        epoch=m['epoch'],
                        train_loss=m['train_loss'],
                        val_loss=m['val_loss'],
                        train_accuracy=m['train_accuracy'],
                        val_accuracy=m['val_accuracy'],
                        learning_rate=m['learning_rate'],
                        timestamp=datetime.fromisoformat(m['timestamp'])
                    ) for m in self.model_metadata['training_history']
                ]
            
            # Load performance metrics if available
            if self.model_metadata.get('model_performance'):
                perf_data = self.model_metadata['model_performance']
                self.model_performance = ModelPerformance(
                    accuracy=perf_data['accuracy'],
                    precision=perf_data['precision'],
                    recall=perf_data['recall'],
                    f1_score=perf_data['f1_score'],
                    confusion_matrix=perf_data['confusion_matrix'],
                    class_distribution=perf_data['class_distribution'],
                    timestamp=datetime.fromisoformat(perf_data['timestamp'])
                )
            
            logger.info(f"Model loaded successfully: version {version}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        try:
            checkpoint_path = os.path.join(self.model_dir, filename)
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
                'config': self.config.to_dict()
            }, checkpoint_path)
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
    
    def _load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        try:
            checkpoint_path = os.path.join(self.model_dir, filename)
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                if self.optimizer and checkpoint.get('optimizer_state_dict'):
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")    def pl
ot_training_history(self, save_path: Optional[str] = None) -> str:
        """
        Plot training history.
        
        Args:
            save_path: Path to save the plot
            
        Returns:
            Path to saved plot
        """
        try:
            if not self.training_history:
                logger.warning("No training history available")
                return ""
            
            # Extract data
            epochs = [m.epoch for m in self.training_history]
            train_losses = [m.train_loss for m in self.training_history]
            val_losses = [m.val_loss for m in self.training_history]
            train_accs = [m.train_accuracy for m in self.training_history]
            val_accs = [m.val_accuracy for m in self.training_history]
            learning_rates = [m.learning_rate for m in self.training_history]
            
            # Create subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Loss plot
            ax1.plot(epochs, train_losses, label='Train Loss', color='blue')
            if any(val_losses):
                ax1.plot(epochs, val_losses, label='Validation Loss', color='red')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training and Validation Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Accuracy plot
            ax2.plot(epochs, train_accs, label='Train Accuracy', color='blue')
            if any(val_accs):
                ax2.plot(epochs, val_accs, label='Validation Accuracy', color='red')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.set_title('Training and Validation Accuracy')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Learning rate plot
            ax3.plot(epochs, learning_rates, color='green')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Learning Rate')
            ax3.set_title('Learning Rate Schedule')
            ax3.set_yscale('log')
            ax3.grid(True, alpha=0.3)
            
            # Confusion matrix (if available)
            if self.model_performance and self.model_performance.confusion_matrix:
                cm = np.array(self.model_performance.confusion_matrix)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4)
                ax4.set_xlabel('Predicted')
                ax4.set_ylabel('Actual')
                ax4.set_title('Confusion Matrix')
                ax4.set_xticklabels(['Down', 'Neutral', 'Up'])
                ax4.set_yticklabels(['Down', 'Neutral', 'Up'])
            else:
                ax4.text(0.5, 0.5, 'No confusion matrix available', 
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Confusion Matrix')
            
            plt.tight_layout()
            
            # Save plot
            if save_path is None:
                save_path = os.path.join(self.model_dir, f'training_history_v{self.model_version}.png')
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Training history plot saved to {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"Error plotting training history: {e}")
            return ""
    
    def plot_predictions(self, df: pd.DataFrame, predictions: List[Dict[str, Any]], 
                        save_path: Optional[str] = None) -> str:
        """
        Plot predictions against actual prices.
        
        Args:
            df: Price dataframe
            predictions: List of predictions
            save_path: Path to save the plot
            
        Returns:
            Path to saved plot
        """
        try:
            if not predictions:
                logger.warning("No predictions to plot")
                return ""
            
            # Prepare data
            prices = df['close'].values
            timestamps = df.index if hasattr(df, 'index') else range(len(prices))
            
            # Extract prediction data
            pred_indices = [p['sequence_index'] + self.config.sequence_length for p in predictions]
            pred_directions = [p['predicted_direction'] for p in predictions]
            pred_confidences = [p['confidence'] for p in predictions]
            
            # Create plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
            
            # Price plot with predictions
            ax1.plot(timestamps, prices, label='Close Price', color='black', linewidth=1)
            
            # Add prediction markers
            for i, (idx, direction, confidence) in enumerate(zip(pred_indices, pred_directions, pred_confidences)):
                if idx < len(prices):
                    color = 'green' if direction == 'up' else 'red' if direction == 'down' else 'gray'
                    marker = '^' if direction == 'up' else 'v' if direction == 'down' else 'o'
                    alpha = min(1.0, confidence * 2)  # Scale alpha by confidence
                    
                    ax1.scatter(timestamps[idx], prices[idx], color=color, marker=marker, 
                              s=50, alpha=alpha, edgecolors='black', linewidth=0.5)
            
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Price')
            ax1.set_title('Price Predictions')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Confidence plot
            ax2.bar(range(len(pred_confidences)), pred_confidences, 
                   color=['green' if d == 'up' else 'red' if d == 'down' else 'gray' 
                          for d in pred_directions], alpha=0.7)
            ax2.set_xlabel('Prediction Index')
            ax2.set_ylabel('Confidence')
            ax2.set_title('Prediction Confidence')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            if save_path is None:
                save_path = os.path.join(self.model_dir, f'predictions_v{self.model_version}.png')
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Predictions plot saved to {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"Error plotting predictions: {e}")
            return ""
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary."""
        try:
            summary = {
                'model_info': {
                    'version': self.model_version,
                    'config': self.config.to_dict(),
                    'total_parameters': sum(p.numel() for p in self.model.parameters()) if self.model else 0,
                    'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad) if self.model else 0
                },
                'training_info': {
                    'total_epochs': len(self.training_history),
                    'best_train_loss': min([m.train_loss for m in self.training_history]) if self.training_history else None,
                    'best_val_loss': min([m.val_loss for m in self.training_history if m.val_loss > 0]) if self.training_history else None,
                    'best_train_accuracy': max([m.train_accuracy for m in self.training_history]) if self.training_history else None,
                    'best_val_accuracy': max([m.val_accuracy for m in self.training_history if m.val_accuracy > 0]) if self.training_history else None
                },
                'performance': self.model_performance.to_dict() if self.model_performance else None,
                'preprocessor_info': {
                    'is_fitted': self.preprocessor.is_fitted,
                    'sequence_length': self.preprocessor.sequence_length,
                    'feature_names': self.preprocessor.feature_names
                },
                'metadata': self.model_metadata
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting model summary: {e}")
            return {'error': str(e)}
    
    def validate_input_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate input data for prediction.
        
        Args:
            df: Input dataframe
            
        Returns:
            Validation results
        """
        try:
            validation_results = {
                'is_valid': True,
                'issues': [],
                'warnings': [],
                'data_info': {}
            }
            
            # Check basic requirements
            if len(df) < self.config.sequence_length:
                validation_results['is_valid'] = False
                validation_results['issues'].append(
                    f"Insufficient data: {len(df)} rows, need at least {self.config.sequence_length}"
                )
            
            # Check required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                validation_results['is_valid'] = False
                validation_results['issues'].append(f"Missing required columns: {missing_columns}")
            
            # Check for NaN values
            nan_columns = df.columns[df.isnull().any()].tolist()
            if nan_columns:
                validation_results['warnings'].append(f"Columns with NaN values: {nan_columns}")
            
            # Check data types
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                    validation_results['warnings'].append(f"Column {col} is not numeric")
            
            # Data info
            validation_results['data_info'] = {
                'total_rows': len(df),
                'date_range': {
                    'start': str(df.index[0]) if hasattr(df, 'index') and len(df) > 0 else None,
                    'end': str(df.index[-1]) if hasattr(df, 'index') and len(df) > 0 else None
                },
                'available_columns': df.columns.tolist(),
                'missing_indicators': [name for name in self.preprocessor.feature_names[5:] if name not in df.columns]
            }
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error validating input data: {e}")
            return {
                'is_valid': False,
                'issues': [f"Validation error: {str(e)}"],
                'warnings': [],
                'data_info': {}
            }
    
    def create_real_time_pipeline(self) -> 'RealTimePipeline':
        """Create real-time inference pipeline."""
        return RealTimePipeline(self)
    
    def cleanup_old_models(self, keep_versions: int = 5) -> int:
        """
        Clean up old model versions.
        
        Args:
            keep_versions: Number of recent versions to keep
            
        Returns:
            Number of models removed
        """
        try:
            # Find all model files
            model_files = []
            for filename in os.listdir(self.model_dir):
                if filename.startswith('lstm_model_v') and filename.endswith('.pth'):
                    version = filename.replace('lstm_model_v', '').replace('.pth', '')
                    model_files.append((version, filename))
            
            # Sort by version (reverse to get newest first)
            model_files.sort(key=lambda x: x[0], reverse=True)
            
            # Remove old versions
            removed_count = 0
            for version, filename in model_files[keep_versions:]:
                try:
                    # Remove model file
                    os.remove(os.path.join(self.model_dir, filename))
                    
                    # Remove associated files
                    for prefix in ['preprocessor_v', 'metadata_v']:
                        associated_file = f"{prefix}{version}.pkl" if prefix.startswith('preprocessor') else f"{prefix}{version}.json"
                        associated_path = os.path.join(self.model_dir, associated_file)
                        if os.path.exists(associated_path):
                            os.remove(associated_path)
                    
                    removed_count += 1
                    logger.info(f"Removed old model version: {version}")
                    
                except Exception as e:
                    logger.warning(f"Error removing model version {version}: {e}")
            
            return removed_count
            
        except Exception as e:
            logger.error(f"Error cleaning up old models: {e}")
            return 0


class RealTimePipeline:
    """Real-time inference pipeline for LSTM price predictor."""
    
    def __init__(self, predictor: LSTMPricePredictor):
        """
        Initialize real-time pipeline.
        
        Args:
            predictor: Trained LSTM predictor
        """
        self.predictor = predictor
        self.data_buffer = deque(maxlen=predictor.config.sequence_length * 2)  # Buffer for streaming data
        self.last_prediction = None
        self.prediction_history = deque(maxlen=100)  # Keep last 100 predictions
        
        logger.info("Real-time pipeline initialized")
    
    def add_data_point(self, data_point: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Add new data point and make prediction if enough data available.
        
        Args:
            data_point: Dictionary with OHLCV and indicator data
            
        Returns:
            Prediction result if available, None otherwise
        """
        try:
            # Add to buffer
            self.data_buffer.append(data_point)
            
            # Check if we have enough data for prediction
            if len(self.data_buffer) >= self.predictor.config.sequence_length:
                # Convert buffer to dataframe
                df = pd.DataFrame(list(self.data_buffer))
                
                # Make prediction
                prediction = self.predictor.predict(df)
                
                if prediction:
                    # Store prediction
                    self.last_prediction = prediction
                    self.prediction_history.append(prediction)
                    
                    return prediction
            
            return None
            
        except Exception as e:
            logger.error(f"Error in real-time prediction: {e}")
            return None
    
    def get_prediction_statistics(self) -> Dict[str, Any]:
        """Get statistics from recent predictions."""
        try:
            if not self.prediction_history:
                return {'error': 'No predictions available'}
            
            predictions = list(self.prediction_history)
            
            # Direction distribution
            directions = [p['predicted_direction'] for p in predictions]
            direction_counts = {direction: directions.count(direction) for direction in ['up', 'down', 'neutral']}
            
            # Confidence statistics
            confidences = [p['confidence'] for p in predictions]
            
            return {
                'total_predictions': len(predictions),
                'direction_distribution': direction_counts,
                'confidence_stats': {
                    'mean': np.mean(confidences),
                    'std': np.std(confidences),
                    'min': np.min(confidences),
                    'max': np.max(confidences)
                },
                'last_prediction': self.last_prediction,
                'buffer_size': len(self.data_buffer)
            }
            
        except Exception as e:
            logger.error(f"Error getting prediction statistics: {e}")
            return {'error': str(e)}
    
    def reset_buffer(self):
        """Reset the data buffer."""
        self.data_buffer.clear()
        logger.info("Real-time pipeline buffer reset")


# Utility functions
def create_sample_data(n_samples: int = 1000, sequence_length: int = 60) -> pd.DataFrame:
    """Create sample data for testing."""
    np.random.seed(42)
    
    # Generate synthetic price data
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='5T')
    
    # Random walk for prices
    price_changes = np.random.normal(0, 0.001, n_samples)
    prices = 100 * np.exp(np.cumsum(price_changes))
    
    # OHLCV data
    opens = prices + np.random.normal(0, 0.1, n_samples)
    highs = np.maximum(opens, prices) + np.abs(np.random.normal(0, 0.2, n_samples))
    lows = np.minimum(opens, prices) - np.abs(np.random.normal(0, 0.2, n_samples))
    volumes = np.random.lognormal(10, 0.5, n_samples)
    
    # Technical indicators (simplified)
    rsi = 50 + 30 * np.sin(np.arange(n_samples) * 0.1) + np.random.normal(0, 5, n_samples)
    rsi = np.clip(rsi, 0, 100)
    
    macd = np.random.normal(0, 0.5, n_samples)
    macd_signal = macd + np.random.normal(0, 0.1, n_samples)
    
    # Moving averages
    sma_20 = pd.Series(prices).rolling(20, min_periods=1).mean().values
    ema_12 = pd.Series(prices).ewm(span=12).mean().values
    ema_26 = pd.Series(prices).ewm(span=26).mean().values
    
    # Bollinger bands
    bb_std = pd.Series(prices).rolling(20, min_periods=1).std().values
    bb_upper = sma_20 + 2 * bb_std
    bb_lower = sma_20 - 2 * bb_std
    
    # ATR (simplified)
    atr = np.random.uniform(0.5, 2.0, n_samples)
    
    # Volume SMA
    volume_sma = pd.Series(volumes).rolling(20, min_periods=1).mean().values
    
    df = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': prices,
        'volume': volumes,
        'rsi': rsi,
        'macd': macd,
        'macd_signal': macd_signal,
        'bb_upper': bb_upper,
        'bb_lower': bb_lower,
        'sma_20': sma_20,
        'ema_12': ema_12,
        'ema_26': ema_26,
        'atr': atr,
        'volume_sma': volume_sma
    }, index=dates)
    
    return df