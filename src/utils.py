"""
Utility functions for model evaluation
"""
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def calculate_rmse(y_true, y_pred):
    """
    Calculate Root Mean Squared Error
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
    
    Returns:
        float: RMSE value
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_mae(y_true, y_pred):
    """
    Calculate Mean Absolute Error
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
    
    Returns:
        float: MAE value
    """
    return mean_absolute_error(y_true, y_pred)

def get_metrics(y_true, y_pred):
    """
    Get both RMSE and MAE metrics
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
    
    Returns:
        dict: Dictionary containing RMSE and MAE
    """
    rmse = calculate_rmse(y_true, y_pred)
    mae = calculate_mae(y_true, y_pred)
    
    return {
        'RMSE': round(rmse, 4),
        'MAE': round(mae, 4)
    }