"""
Data processing module for stock price prediction
Handles data loading, preprocessing, and train-test split
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime

class DataProcessor:
    """
    Handles all data preprocessing operations
    """
    
    def __init__(self, data_path='data/MicrosoftStock.csv'):
        """
        Initialize DataProcessor
        
        Args:
            data_path: Path to the CSV file
        """
        self.data_path = data_path
        self.data = None
        self.scaler = StandardScaler()
        self.scaled_data = None
        self.training_data_len = None
        
    def load_data(self):
        """
        Load the stock data from CSV
        
        Returns:
            DataFrame: Loaded stock data
        """
        self.data = pd.read_csv(self.data_path)
        self.data['date'] = pd.to_datetime(self.data['date'])
        print(f"✅ Data loaded successfully! Shape: {self.data.shape}")
        return self.data
    
    def get_close_prices(self):
        """
        Extract close prices for prediction
        
        Returns:
            numpy array: Close price values
        """
        stock_close = self.data.filter(["close"])
        dataset = stock_close.values
        return dataset
    
    def scale_data(self, dataset, train_ratio=0.95):
        """
        scale the data WITHOUT leakage.
        Fit scaler ONLY on training data.
        """
        # Split index first
        self.training_data_len = int(np.ceil(len(dataset) * train_ratio))
        
        # Split raw data
        train_data = dataset[:self.training_data_len]
        test_data = dataset[self.training_data_len:]
        
        # Fit scaler ONLY on training data
        self.scaler.fit(train_data)
        
        # Transform both
        train_scaled = self.scaler.transform(train_data)
        test_scaled = self.scaler.transform(test_data)
        
        # Combine back in correct order
        self.scaled_data = np.vstack((train_scaled, test_scaled))
        
        print(f"✅ Data scaled properly (NO leakage)!")
        print(f"   Training samples: {self.training_data_len}")
        
        return self.scaled_data, self.training_data_len

    
    def create_lstm_sequences(self, sequence_length=60):
        """
        Create sliding window sequences for LSTM
        
        Args:
            sequence_length: Number of days to look back (default 60)
        
        Returns:
            tuple: (x_train, y_train, x_test, y_test, test_data_original)
        """
        # Training data
        training_data = self.scaled_data[0:self.training_data_len]
        
        x_train, y_train = [], []
        
        for i in range(sequence_length, len(training_data)):
            x_train.append(training_data[i-sequence_length:i, 0])
            y_train.append(training_data[i, 0])
        
        x_train, y_train = np.array(x_train), np.array(y_train)
        
        # Reshape for LSTM [samples, time steps, features]
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        
        # Test data
        train_scaled = self.scaled_data[:self.training_data_len]
        test_scaled = self.scaled_data[self.training_data_len:]

        # Combine last sequence_length from train for continuity
        test_data = np.vstack((
            train_scaled[-sequence_length:],
            test_scaled
        ))

        dataset = self.get_close_prices()
        
        x_test, y_test = [], dataset[self.training_data_len:]
        
        for i in range(sequence_length, len(test_data)):
            x_test.append(test_data[i-sequence_length:i, 0])
        
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        
        print(f"✅ Sequences created!")
        print(f"   Training samples: {x_train.shape[0]}")
        print(f"   Test samples: {x_test.shape[0]}")
        
        return x_train, y_train, x_test, y_test
    
    def inverse_transform(self, predictions):
        """
        Convert scaled predictions back to original price scale
        
        Args:
            predictions: Scaled prediction values
        
        Returns:
            numpy array: Original scale predictions
        """
        return self.scaler.inverse_transform(predictions)
    
    def get_train_test_data(self):
        """
        Get train and test dataframes with dates
        
        Returns:
            tuple: (train_df, test_df)
        """
        train = self.data[:self.training_data_len].copy()
        test = self.data[self.training_data_len:].copy()
        
        return train, test
    
    def create_xgboost_features(self, lookback=60):
        """
        Create features for XGBoost using a lookback window approach
        Similar to LSTM but flattened for tree-based models
        
        Args:
            lookback: Number of previous days to include as features
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        df = self.data.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        # Create lag features (previous N days of prices)
        for i in range(1, lookback + 1):
            df[f'Close_Lag_{i}'] = df['close'].shift(i)
        
        # Add simple technical indicators
        df['MA_5'] = df['close'].rolling(window=5).mean()
        df['MA_20'] = df['close'].rolling(window=20).mean()
        df['Volatility_10'] = df['close'].rolling(window=10).std()
        
        # RSI
        def calculate_rsi(data, window=14):
            delta = data.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        df['RSI'] = calculate_rsi(df['close'])
        
        # Drop NaN rows
        df = df.dropna().reset_index(drop=True)
        
        # Feature columns: last 10 days + technical indicators
        feature_columns = [f'Close_Lag_{i}' for i in range(1, lookback + 1)]
        feature_columns += ['MA_5', 'MA_20', 'Volatility_10', 'RSI']
        
        X = df[feature_columns].values
        y = df['close'].values
        
        # Train-test split
        train_size = int(len(X) * 0.95)
        
        X_train = X[:train_size]
        X_test = X[train_size:]
        y_train = y[:train_size]
        y_test = y[train_size:]
        
        # NO SCALING - XGBoost often works better without it for price data
        
        print(f"✅ XGBoost features created!")
        print(f"   Features: {len(feature_columns)}")
        print(f"   Lookback window: {lookback} days")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Test samples: {len(X_test)}")
        
        # Store for later
        self.xgboost_df = df
        self.xgboost_train_size = train_size
        self.xgb_feature_names = feature_columns
        
        return X_train, X_test, y_train, y_test

    def get_xgboost_train_test_data(self):
        """
        Get train and test dataframes for XGBoost (with dates)
        
        Returns:
            tuple: (train_df, test_df)
        """
        if not hasattr(self, 'xgboost_df'):
            raise ValueError("Call create_xgboost_features() first!")
        
        train = self.xgboost_df[:self.xgboost_train_size].copy()
        test = self.xgboost_df[self.xgboost_train_size:].copy()
        
        return train, test