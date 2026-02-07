"""
Model Engine module containing LSTM and XGBoost model classes
"""
from tensorflow import keras
import numpy as np
import joblib
import os

class LSTMModel:
    """
    LSTM Model for stock price prediction
    """
    
    def __init__(self, sequence_length=60):
        """
        Initialize LSTM Model
        
        Args:
            sequence_length: Number of days to look back (default 60)
        """
        self.sequence_length = sequence_length
        self.model = None
        self.history = None
        
    def build_model(self, input_shape):
        """
        Build the LSTM architecture
        
        Args:
            input_shape: Shape of input data (sequence_length, 1)
        """
        self.model = keras.models.Sequential()
        
        # First LSTM layer
        self.model.add(keras.layers.LSTM(
            64, 
            return_sequences=True, 
            input_shape=input_shape
        ))
        
        # Second LSTM layer
        self.model.add(keras.layers.LSTM(64, return_sequences=False))
        
        # Dense layer
        self.model.add(keras.layers.Dense(128, activation='relu'))
        
        # Dropout layer
        self.model.add(keras.layers.Dropout(0.3))
        
        # Output layer
        self.model.add(keras.layers.Dense(1))
        
        # Compile the model
        self.model.compile(
            optimizer='adam',
            loss='mae',
            metrics=[keras.metrics.RootMeanSquaredError()]
        )
        
        print("✅ LSTM Model built successfully!")
        self.model.summary()
        
        return self.model
    
    def train(self, x_train, y_train, epochs=20, batch_size=32):
        """
        Train the LSTM model
        
        Args:
            x_train: Training sequences
            y_train: Training targets
            epochs: Number of training epochs (default 20)
            batch_size: Batch size (default 32)
        
        Returns:
            History object
        """
        if self.model is None:
            # Build model automatically if not built
            self.build_model((x_train.shape[1], 1))
        
        print(f"\n🚀 Training LSTM model for {epochs} epochs...")
        
        self.history = self.model.fit(
            x_train, 
            y_train, 
            batch_size=batch_size, 
            epochs=epochs,
            verbose=1
        )
        
        print("✅ Training completed!")
        return self.history
    
    def predict(self, x_test):
        """
        Make predictions using the trained model
        
        Args:
            x_test: Test sequences
        
        Returns:
            numpy array: Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained! Call train() first.")
        
        predictions = self.model.predict(x_test)
        return predictions
    
    def save_model(self, filepath='models/saved_models/lstm_model.h5'):
        """
        Save the trained model
        
        Args:
            filepath: Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        self.model.save(filepath)
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath='models/saved_models/lstm_model.h5'):
        """
        Load a saved model
        
        Args:
            filepath: Path to the saved model
        """
        self.model = keras.models.load_model(filepath)
        print(f"✅ Model loaded from {filepath}")
        return self.model
    
    def get_model_info(self):
        """
        Get model architecture information
        
        Returns:
            dict: Model information
        """
        return {
            'Model Type': 'LSTM (Long Short-Term Memory)',
            'Architecture': '2 LSTM Layers (64 units each) + Dense Layer (128 units)',
            'Sequence Length': f'{self.sequence_length} days',
            'Optimizer': 'Adam',
            'Loss Function': 'MAE (Mean Absolute Error)',
            'Dropout': '0.3',
            'Total Parameters': self.model.count_params() if self.model else 'N/A'
        }


# Placeholder for XGBoost (we'll add this tomorrow)
class XGBoostModel:
    """
    XGBoost Model for stock price prediction
    (To be implemented)
    """
    
    def __init__(self):
        print("⏳ XGBoost model - Coming soon!")
        pass