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
        Scale the data using StandardScaler
        
        Args:
            dataset: Raw price data
            train_ratio: Percentage of data for training (default 95%)
        
        Returns:
            tuple: (scaled_data, training_data_length)
        """
        self.training_data_len = int(np.ceil(len(dataset) * train_ratio))
        self.scaled_data = self.scaler.fit_transform(dataset)
        
        print(f"✅ Data scaled! Training samples: {self.training_data_len}")
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
        test_data = self.scaled_data[self.training_data_len - sequence_length:]
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
    
    def create_xgboost_features(self):
        """
        Create engineered features for XGBoost model
        Includes: Moving Averages, RSI, Time features
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        df = self.data.copy()
        
        # Sort by date to ensure chronological order
        df = df.sort_values('date').reset_index(drop=True)
        
        # 1. Moving Averages
        df['MA_7'] = df['close'].rolling(window=7).mean()
        df['MA_21'] = df['close'].rolling(window=21).mean()
        df['MA_50'] = df['close'].rolling(window=50).mean()
        
        # 2. Moving Average Ratios (better than absolute values)
        df['MA_Ratio_7_21'] = df['MA_7'] / df['MA_21']
        df['MA_Ratio_21_50'] = df['MA_21'] / df['MA_50']
        
        # 3. RSI (Relative Strength Index)
        def calculate_rsi(data, window=14):
            delta = data.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        df['RSI'] = calculate_rsi(df['close'])
        
        # 4. Time-based features
        df['Day'] = df['date'].dt.dayofweek
        df['Month'] = df['date'].dt.month
        
        # 5. Lag features - MORE of them
        for i in range(1, 6):  # Last 5 days
            df[f'Lag_{i}'] = df['close'].shift(i)
        
        # 6. Price change percentage (more stable than absolute)
        df['Price_Change_Pct_1d'] = df['close'].pct_change(1) * 100
        df['Price_Change_Pct_5d'] = df['close'].pct_change(5) * 100
        
        # 7. Volatility
        df['Volatility_10'] = df['close'].rolling(window=10).std()
        df['Volatility_30'] = df['close'].rolling(window=30).std()
        
        # Drop rows with NaN values
        df = df.dropna().reset_index(drop=True)
        
        # ✅ IMPROVED: Use percentage-based features instead of absolute prices
        feature_columns = [
            'MA_Ratio_7_21', 'MA_Ratio_21_50',  # Ratios instead of absolute MA
            'RSI', 
            'Day', 'Month',
            'Lag_1', 'Lag_2', 'Lag_3', 'Lag_4', 'Lag_5',
            'Price_Change_Pct_1d', 'Price_Change_Pct_5d',
            'Volatility_10', 'Volatility_30'
        ]
        
        X = df[feature_columns].values
        y = df['close'].values
        
        # Train-test split
        train_size = int(len(X) * 0.95)
        
        X_train = X[:train_size]
        X_test = X[train_size:]
        y_train = y[:train_size]
        y_test = y[train_size:]
        
        # Feature scaling
        from sklearn.preprocessing import StandardScaler
        self.xgb_scaler = StandardScaler()
        X_train = self.xgb_scaler.fit_transform(X_train)
        X_test = self.xgb_scaler.transform(X_test)
        
        print(f"✅ XGBoost features created and scaled!")
        print(f"   Features: {len(feature_columns)}")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Test samples: {len(X_test)}")
        
        # Store for later use
        self.xgboost_df = df
        self.xgboost_train_size = train_size
        
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