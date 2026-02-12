"""
Visualization module for creating graphs and charts
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class Visualizer:
    """
    Handles all visualization tasks for stock price predictions
    """
    
    def __init__(self):
        """
        Initialize Visualizer with styling
        """
        # Set style for better-looking plots
        plt.style.use('seaborn-v0_8-darkgrid')
        
    def plot_lstm_predictions(self, train_data, test_data, predictions, title="LSTM Stock Price Prediction"):
        """
        Create a plot showing actual vs predicted prices for LSTM
        
        Args:
            train_data: DataFrame with training data (must have 'date' and 'close' columns)
            test_data: DataFrame with test data (must have 'date' and 'close' columns)
            predictions: Array of predicted prices
            title: Plot title
        
        Returns:
            matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Plot training data
        ax.plot(
            train_data['date'], 
            train_data['close'], 
            label='Training Data (Actual)', 
            color='#1f77b4',
            linewidth=2
        )
        
        # Plot test data (actual)
        ax.plot(
            test_data['date'], 
            test_data['close'], 
            label='Test Data (Actual)', 
            color='#ff7f0e',
            linewidth=2
        )
        
        # Plot predictions
        ax.plot(
            test_data['date'], 
            predictions, 
            label='Predictions', 
            color='#d62728',
            linewidth=2,
            linestyle='--'
        )
        
        # Styling
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Close Price (USD)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='upper left', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
    
    def plot_comparison(self, train_data, test_data, lstm_predictions, xgboost_predictions=None, 
                       title="Model Comparison: LSTM vs XGBoost"):
        """
        Create a comparison plot for LSTM and XGBoost predictions
        
        Args:
            train_data: DataFrame with training data
            test_data: DataFrame with test data
            lstm_predictions: LSTM model predictions
            xgboost_predictions: XGBoost model predictions (optional)
            title: Plot title
        
        Returns:
            matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Plot training data
        ax.plot(
            train_data['date'], 
            train_data['close'], 
            label='Training Data (Actual)', 
            color='#1f77b4',
            linewidth=2,
            alpha=0.7
        )
        
        # Plot test data (actual)
        ax.plot(
            test_data['date'], 
            test_data['close'], 
            label='Test Data (Actual)', 
            color='#2ca02c',
            linewidth=2.5
        )
        
        # Plot LSTM predictions
        ax.plot(
            test_data['date'], 
            lstm_predictions, 
            label='LSTM Predictions', 
            color='#d62728',
            linewidth=2,
            linestyle='--'
        )
        
        # Plot XGBoost predictions if available
        if xgboost_predictions is not None:
            ax.plot(
                test_data['date'], 
                xgboost_predictions, 
                label='XGBoost Predictions', 
                color='#9467bd',
                linewidth=2,
                linestyle='-.'
            )
        
        # Styling
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Close Price (USD)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='upper left', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
    
    def plot_training_history(self, history):

        """
        Plot training loss history for LSTM
        
        Args:
            history: Keras History object from model.fit()
        
        Returns:
            matplotlib figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot loss
        ax1.plot(history.history['loss'], color='#1f77b4', linewidth=2)
        ax1.set_title('Model Loss During Training', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=11)
        ax1.set_ylabel('Loss (MAE)', fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Plot RMSE
        ax2.plot(history.history['root_mean_squared_error'], color='#ff7f0e', linewidth=2)
        ax2.set_title('Model RMSE During Training', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch', fontsize=11)
        ax2.set_ylabel('RMSE', fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self, feature_importance, feature_names):
        """
        Plot feature importance for XGBoost
        
        Args:
            feature_importance: Array of importance scores
            feature_names: List of feature names
        
        Returns:
            matplotlib figure object
        """
        import pandas as pd
        
        # Create dataframe for sorting
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=True)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Horizontal bar chart
        ax.barh(importance_df['Feature'], importance_df['Importance'], color='#9467bd')
        ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Features', fontsize=12, fontweight='bold')
        ax.set_title('XGBoost Feature Importance', fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        return fig