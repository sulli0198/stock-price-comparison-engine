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