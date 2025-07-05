# predictive-model/deep_learning_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import ta
import joblib
import logging
from datetime import datetime
import redis
import psycopg2
from typing import Dict, List, Tuple, Optional
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingDataset(Dataset):
    """Custom dataset for trading data with advanced preprocessing"""
    
    def __init__(self, data: pd.DataFrame, sequence_length: int = 60, 
                 target_lookahead: int = 5, features: List[str] = None):
        self.data = data
        self.sequence_length = sequence_length
        self.target_lookahead = target_lookahead
        self.features = features or self._get_default_features()
        
        # Prepare sequences
        self.sequences, self.targets = self._prepare_sequences()
        
    def _get_default_features(self) -> List[str]:
        """Get default feature set"""
        return [
            'open', 'high', 'low', 'close', 'volume',
            'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_middle', 'bb_lower',
            'ema_12', 'ema_26', 'ema_50', 'ema_200',
            'atr', 'adx', 'cci', 'mfi', 'obv',
            'vwap', 'pvt', 'eom', 'fi', 'vpt'
        ]
    
    def _prepare_sequences(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM training"""
        sequences = []
        targets = []
        
        # Calculate all technical indicators
        self._add_technical_indicators()
        
        # Normalize features
        scaler = RobustScaler()
        scaled_features = scaler.fit_transform(self.data[self.features])
        
        # Create sequences
        for i in range(self.sequence_length, len(self.data) - self.target_lookahead):
            seq = scaled_features[i-self.sequence_length:i]
            
            # Multi-target: predict price movement and volatility
            future_returns = (
                self.data['close'].iloc[i:i+self.target_lookahead].pct_change().mean()
            )
            future_volatility = (
                self.data['close'].iloc[i:i+self.target_lookahead].pct_change().std()
            )
            
            # Binary classification target (price up/down)
            price_up = 1 if future_returns > 0.001 else 0  # 0.1% threshold
            
            # Volatility regime (low/medium/high)
            vol_regime = 0 if future_volatility < 0.01 else (1 if future_volatility < 0.02 else 2)
            
            sequences.append(seq)
            targets.append([price_up, vol_regime, future_returns, future_volatility])
            
        return np.array(sequences), np.array(targets)
    
    def _add_technical_indicators(self):
        """Add all technical indicators to the dataset"""
        # Price-based indicators
        self.data['rsi'] = ta.momentum.RSIIndicator(self.data['close']).rsi()
        
        # MACD
        macd = ta.trend.MACD(self.data['close'])
        self.data['macd'] = macd.macd()
        self.data['macd_signal'] = macd.macd_signal()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(self.data['close'])
        self.data['bb_upper'] = bb.bollinger_hband()
        self.data['bb_middle'] = bb.bollinger_mavg()
        self.data['bb_lower'] = bb.bollinger_lband()
        
        # EMAs
        for period in [12, 26, 50, 200]:
            self.data[f'ema_{period}'] = ta.trend.EMAIndicator(
                self.data['close'], window=period
            ).ema_indicator()
        
        # ATR (Average True Range)
        self.data['atr'] = ta.volatility.AverageTrueRange(
            self.data['high'], self.data['low'], self.data['close']
        ).average_true_range()
        
        # ADX (Average Directional Index)
        self.data['adx'] = ta.trend.ADXIndicator(
            self.data['high'], self.data['low'], self.data['close']
        ).adx()
        
        # CCI (Commodity Channel Index)
        self.data['cci'] = ta.trend.CCIIndicator(
            self.data['high'], self.data['low'], self.data['close']
        ).cci()
        
        # MFI (Money Flow Index)
        self.data['mfi'] = ta.volume.MFIIndicator(
            self.data['high'], self.data['low'], self.data['close'], self.data['volume']
        ).money_flow_index()
        
        # OBV (On Balance Volume)
        self.data['obv'] = ta.volume.OnBalanceVolumeIndicator(
            self.data['close'], self.data['volume']
        ).on_balance_volume()
        
        # VWAP
        self.data['vwap'] = ta.volume.VolumeWeightedAveragePrice(
            self.data['high'], self.data['low'], self.data['close'], self.data['volume']
        ).volume_weighted_average_price()
        
        # Additional volume indicators
        self.data['pvt'] = ta.volume.PutCallVolumeRatio(
            self.data['close'], self.data['volume']
        ).put_call_volume_ratio()
        
        self.data['eom'] = ta.volume.EaseOfMovementIndicator(
            self.data['high'], self.data['low'], self.data['volume']
        ).ease_of_movement()
        
        self.data['fi'] = ta.volume.ForceIndexIndicator(
            self.data['close'], self.data['volume']
        ).force_index()
        
        self.data['vpt'] = ta.volume.VolumePriceTrendIndicator(
            self.data['close'], self.data['volume']
        ).volume_price_trend()
        
        # Fill NaN values
        self.data.fillna(method='ffill', inplace=True)
        self.data.fillna(0, inplace=True)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.sequences[idx]),
            torch.FloatTensor(self.targets[idx])
        )

class AttentionLSTM(nn.Module):
    """Advanced LSTM with Multi-Head Attention and Residual Connections"""
    
    def __init__(self, input_size: int, hidden_size: int = 256, 
                 num_layers: int = 4, num_heads: int = 8, 
                 dropout: float = 0.2):
        super(AttentionLSTM, self).__init__()
        
        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Bidirectional LSTM layers
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(
                input_size=hidden_size if i == 0 else hidden_size * 2,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if i < num_layers - 1 else 0
            ) for i in range(num_layers)
        ])
        
        # Multi-head attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_size * 2,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size * 2) for _ in range(num_layers)
        ])
        
        # Position encoding
        self.positional_encoding = self._create_positional_encoding(1000, hidden_size * 2)
        
        # Output layers for different predictions
        self.price_direction_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.volatility_regime_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # 3 volatility regimes
            nn.Softmax(dim=-1)
        )
        
        self.return_prediction_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.volatility_prediction_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()  # Ensure positive volatility
        )
        
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encoding"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.positional_encoding[:, :seq_len, :x.size(-1)].to(x.device)
        
        # Process through LSTM and attention layers
        for i, (lstm, attention, norm) in enumerate(
            zip(self.lstm_layers, self.attention_layers, self.layer_norms)
        ):
            # LSTM forward pass
            lstm_out, _ = lstm(x)
            
            # Multi-head attention with residual connection
            attn_out, _ = attention(lstm_out, lstm_out, lstm_out)
            
            # Residual connection and layer normalization
            x = norm(attn_out + lstm_out)
        
        # Global average pooling and max pooling
        avg_pool = torch.mean(x, dim=1)
        max_pool, _ = torch.max(x, dim=1)
        
        # Concatenate pooled features
        global_features = torch.cat([avg_pool, max_pool], dim=-1)
        
        # Also use the last timestep
        last_timestep = x[:, -1, :]
        
        # Combine global and temporal features
        combined_features = (global_features + last_timestep) / 2
        
        # Generate predictions
        price_direction = self.price_direction_head(combined_features)
        volatility_regime = self.volatility_regime_head(combined_features)
        return_pred = self.return_prediction_head(combined_features)
        volatility_pred = self.volatility_prediction_head(combined_features)
        
        return price_direction, volatility_regime, return_pred, volatility_pred

class DeepLearningTrainer:
    """Complete training pipeline for the deep learning model"""
    
    def __init__(self, model_save_path: str = './models', 
                 redis_host: str = 'localhost', redis_port: int = 6379,
                 db_config: Dict = None):
        self.model_save_path = model_save_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = RobustScaler()
        
        # Redis connection for caching
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        
        # Database connection for storing results
        self.db_config = db_config or {
            'host': 'localhost',
            'database': 'trading',
            'user': 'trader',
            'password': 'password'
        }
        
    def train(self, data: pd.DataFrame, epochs: int = 100, 
              batch_size: int = 32, learning_rate: float = 0.001,
              validation_split: float = 0.2):
        """Train the deep learning model"""
        
        logger.info("Starting deep learning model training...")
        
        # Create dataset
        dataset = TradingDataset(data)
        
        # Split data
        train_size = int((1 - validation_split) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )
        
        # Initialize model
        input_size = len(dataset.features)
        self.model = AttentionLSTM(input_size=input_size).to(self.device)
        
        # Loss functions
        bce_loss = nn.BCELoss()
        ce_loss = nn.CrossEntropyLoss()
        mse_loss = nn.MSELoss()
        mae_loss = nn.L1Loss()
        
        # Optimizer with learning rate scheduling
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=0.01
        )
        
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate * 10,
            epochs=epochs,
            steps_per_epoch=len(train_loader)
        )
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'learning_rates': []
        }
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (sequences, targets) in enumerate(train_loader):
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                price_pred, vol_regime_pred, return_pred, vol_pred = self.model(sequences)
                
                # Calculate losses
                price_loss = bce_loss(price_pred.squeeze(), targets[:, 0])
                vol_regime_loss = ce_loss(vol_regime_pred, targets[:, 1].long())
                return_loss = mse_loss(return_pred.squeeze(), targets[:, 2])
                vol_loss = mae_loss(vol_pred.squeeze(), targets[:, 3])
                
                # Combined loss with different weights
                total_loss = (
                    0.4 * price_loss + 
                    0.2 * vol_regime_loss + 
                    0.3 * return_loss + 
                    0.1 * vol_loss
                )
                
                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                
                # Track metrics
                train_loss += total_loss.item()
                predictions = (price_pred.squeeze() > 0.5).float()
                train_correct += (predictions == targets[:, 0]).sum().item()
                train_total += targets.size(0)
                
                # Log progress
                if batch_idx % 10 == 0:
                    logger.info(
                        f'Epoch: {epoch}/{epochs}, Batch: {batch_idx}/{len(train_loader)}, '
                        f'Loss: {total_loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}'
                    )
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for sequences, targets in val_loader:
                    sequences = sequences.to(self.device)
                    targets = targets.to(self.device)
                    
                    # Forward pass
                    price_pred, vol_regime_pred, return_pred, vol_pred = self.model(sequences)
                    
                    # Calculate losses
                    price_loss = bce_loss(price_pred.squeeze(), targets[:, 0])
                    vol_regime_loss = ce_loss(vol_regime_pred, targets[:, 1].long())
                    return_loss = mse_loss(return_pred.squeeze(), targets[:, 2])
                    vol_loss = mae_loss(vol_pred.squeeze(), targets[:, 3])
                    
                    total_loss = (
                        0.4 * price_loss + 
                        0.2 * vol_regime_loss + 
                        0.3 * return_loss + 
                        0.1 * vol_loss
                    )
                    
                    val_loss += total_loss.item()
                    predictions = (price_pred.squeeze() > 0.5).float()
                    val_correct += (predictions == targets[:, 0]).sum().item()
                    val_total += targets.size(0)
            
            # Calculate epoch metrics
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_accuracy = train_correct / train_total
            val_accuracy = val_correct / val_total
            
            # Update history
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['train_accuracy'].append(train_accuracy)
            history['val_accuracy'].append(val_accuracy)
            history['learning_rates'].append(scheduler.get_last_lr()[0])
            
            # Log epoch results
            logger.info(
                f'Epoch {epoch}/{epochs} - '
                f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
                f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}'
            )
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                self.save_model(epoch, best_val_loss)
                logger.info(f'New best model saved with val_loss: {best_val_loss:.4f}')
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= patience:
                logger.info(f'Early stopping triggered after {epoch} epochs')
                break
            
            # Store metrics in Redis
            self._cache_training_metrics(epoch, history)
        
        # Save final model and history
        self._save_training_history(history)
        
        return history
    
    def predict(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Make predictions using the trained model"""
        
        if self.model is None:
            self.load_model()
        
        self.model.eval()
        
        # Prepare data
        dataset = TradingDataset(data)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        all_predictions = {
            'price_direction': [],
            'volatility_regime': [],
            'return_prediction': [],
            'volatility_prediction': []
        }
        
        with torch.no_grad():
            for sequences, _ in loader:
                sequences = sequences.to(self.device)
                
                price_pred, vol_regime_pred, return_pred, vol_pred = self.model(sequences)
                
                all_predictions['price_direction'].append(price_pred.cpu().numpy())
                all_predictions['volatility_regime'].append(vol_regime_pred.cpu().numpy())
                all_predictions['return_prediction'].append(return_pred.cpu().numpy())
                all_predictions['volatility_prediction'].append(vol_pred.cpu().numpy())
        
        # Concatenate all predictions
        for key in all_predictions:
            all_predictions[key] = np.concatenate(all_predictions[key], axis=0)
        
        return all_predictions
    
    def save_model(self, epoch: int, loss: float):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'loss': loss,
            'model_config': {
                'input_size': self.model.input_projection[0].in_features,
                'hidden_size': 256,
                'num_layers': 4,
                'num_heads': 8,
                'dropout': 0.2
            }
        }
        
        torch.save(checkpoint, f'{self.model_save_path}/model_checkpoint.pt')
        
        # Also save scaler
        joblib.dump(self.scaler, f'{self.model_save_path}/scaler.pkl')
    
    def load_model(self):
        """Load saved model"""
        checkpoint = torch.load(f'{self.model_save_path}/model_checkpoint.pt')
        
        config = checkpoint['model_config']
        self.model = AttentionLSTM(**config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.scaler = joblib.load(f'{self.model_save_path}/scaler.pkl')
        
        logger.info(f"Model loaded from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.4f}")
    
    def _cache_training_metrics(self, epoch: int, history: Dict):
        """Cache training metrics in Redis"""
        metrics = {
            'epoch': epoch,
            'train_loss': history['train_loss'][-1],
            'val_loss': history['val_loss'][-1],
            'train_accuracy': history['train_accuracy'][-1],
            'val_accuracy': history['val_accuracy'][-1],
            'timestamp': datetime.now().isoformat()
        }
        
        self.redis_client.hset(
            'training_metrics',
            f'epoch_{epoch}',
            json.dumps(metrics)
        )
        
        # Also store latest metrics
        self.redis_client.set('latest_training_metrics', json.dumps(metrics))
    
    def _save_training_history(self, history: Dict):
        """Save complete training history to database"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            
            # Create table if not exists
            cur.execute("""
                CREATE TABLE IF NOT EXISTS training_history (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    model_type VARCHAR(50),
                    epochs INTEGER,
                    best_val_loss FLOAT,
                    best_val_accuracy FLOAT,
                    history JSONB
                )
            """)
            
            # Insert training history
            cur.execute("""
                INSERT INTO training_history (model_type, epochs, best_val_loss, best_val_accuracy, history)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                'AttentionLSTM',
                len(history['train_loss']),
                min(history['val_loss']),
                max(history['val_accuracy']),
                json.dumps(history)
            ))
            
            conn.commit()
            cur.close()
            conn.close()
            
            logger.info("Training history saved to database")
            
        except Exception as e:
            logger.error(f"Error saving training history: {str(e)}")

__all__ = ['DeepLearningTrainer']

# Usage example
if __name__ == "__main__":
    # Load your data
    data = pd.read_csv('historical_data.csv')
    
    # Initialize trainer
    trainer = DeepLearningTrainer(
        model_save_path='./models',
        redis_host='localhost',
        redis_port=6379
    )
    
    # Train model
    history = trainer.train(
        data,
        epochs=100,
        batch_size=32,
        learning_rate=0.001
    )
    
    # Make predictions
    predictions = trainer.predict(data[-100:])
    
    print(f"Price direction probability: {predictions['price_direction'][-1][0]:.4f}")
    print(f"Volatility regime: {predictions['volatility_regime'][-1].argmax()}")