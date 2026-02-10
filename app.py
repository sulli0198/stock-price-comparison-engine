"""
Streamlit Dashboard for Stock Price Prediction
LSTM Model Implementation
"""
import streamlit as st
import pandas as pd
import numpy as np
from src.data_processor import DataProcessor
from src.model_engine import LSTMModel
from src.visualizer import Visualizer
from src.utils import get_metrics
from src.model_engine import XGBoostModel
import os

# Page configuration
st.set_page_config(
    page_title="Stock Price Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for attractive UI
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f0f2f6;
        border-radius: 5px;
        font-weight: bold;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">📈 Stock Price Prediction System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Comparing LSTM and XGBoost Models for Microsoft Stock Prediction</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/stock-market.png", width=100)
    st.title("⚙️ Model Settings")
    st.markdown("---")
    
    # Model parameters
    st.subheader("LSTM Parameters")
    epochs = st.slider("Training Epochs", min_value=10, max_value=50, value=25, step=5)
    batch_size = st.slider("Batch Size", min_value=16, max_value=64, value=32, step=16)
    
    st.markdown("---")

    st.subheader("XGBoost Parameters")
    n_estimators = st.slider("N Estimators", min_value=50, max_value=500, value=200, step=50)
    max_depth = st.slider("Max Depth", min_value=3, max_value=15, value=7, step=1)
    learning_rate = st.slider("Learning Rate", min_value=0.01, max_value=0.3, value=0.05, step=0.01)

    st.subheader("About")
    st.info("""
    This dashboard demonstrates stock price prediction using:
    - **LSTM**: Deep learning model for sequential data
    - **XGBoost**: Tree-based ML model (Coming Soon)
    
    **Dataset**: Microsoft Stock Historical Data
    """)

# Initialize components
@st.cache_resource
def load_and_prepare_data():
    """Load and prepare data (cached for performance)"""
    processor = DataProcessor('data/MicrosoftStock.csv')
    processor.load_data()
    dataset = processor.get_close_prices()
    processor.scale_data(dataset)
    return processor

# Load data
with st.spinner("🔄 Loading data..."):
    processor = load_and_prepare_data()

st.success("✅ Data loaded successfully!")

# Display dataset info
with st.expander("📊 View Dataset Information"):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(processor.data))
    with col2:
        st.metric("Training Samples", processor.training_data_len)
    with col3:
        st.metric("Test Samples", len(processor.data) - processor.training_data_len)
    with col4:
        st.metric("Features", "Close Price")
    
    st.dataframe(processor.data.head(10), use_container_width=True)

# Tabs for different views
tab1, tab2, tab3 = st.tabs(["🔵 LSTM Model", "🟣 XGBoost Model", "⚖️ Comparison"])

# ===========================
# TAB 1: LSTM MODEL
# ===========================
with tab1:
    st.header("🔵 LSTM Model - Long Short-Term Memory Network")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### About LSTM Model
        LSTM (Long Short-Term Memory) is a type of Recurrent Neural Network (RNN) designed to learn 
        from sequential data. It's particularly effective for time-series prediction like stock prices.
        
        **Key Features:**
        - Uses 60-day sliding window
        - 2 LSTM layers with 64 units each
        - Dense layer with 128 neurons
        - Dropout layer (0.3) to prevent overfitting
        """)
    
    with col2:
        st.markdown("""
        ### Model Architecture
        - **Input**: 60 days of stock prices
        - **LSTM Layer 1**: 64 units
        - **LSTM Layer 2**: 64 units
        - **Dense Layer**: 128 units
        - **Dropout**: 0.3
        - **Output**: Next day price prediction
        """)
    
    st.markdown("---")
    
    # Train button
    if st.button("🚀 Train LSTM Model", type="primary", use_container_width=True):
        
        with st.spinner("🔄 Preparing sequences..."):
            # Create sequences
            x_train, y_train, x_test, y_test = processor.create_lstm_sequences(sequence_length=60)
        
        # Initialize model
        lstm_model = LSTMModel(sequence_length=60)
        
        # Training
        with st.spinner(f"🤖 Training LSTM model for {epochs} epochs... This may take a few minutes."):
            progress_bar = st.progress(0)
            
            # Build and train
            lstm_model.build_model((x_train.shape[1], 1))
            history = lstm_model.train(x_train, y_train, epochs=epochs, batch_size=batch_size)
            
            progress_bar.progress(100)
        
        st.success("✅ Training completed!")
        
        # Make predictions
        with st.spinner("🔮 Making predictions..."):
            predictions_scaled = lstm_model.predict(x_test)
            predictions = processor.inverse_transform(predictions_scaled)
        
        # Calculate metrics
        metrics = get_metrics(y_test, predictions)
        
        # Display metrics
        st.subheader("📊 Model Performance")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                label="Root Mean Squared Error (RMSE)",
                value=f"{metrics['RMSE']:.2f}",
                delta=None
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                label="Mean Absolute Error (MAE)",
                value=f"{metrics['MAE']:.2f}",
                delta=None
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.info(f"""
        **Interpretation:**
        - RMSE of ${metrics['RMSE']:.2f} means the model's predictions are off by about ${metrics['RMSE']:.2f} on average.
        - MAE of ${metrics['MAE']:.2f} shows the average absolute difference between predicted and actual prices.
        """)
        
        # Visualizations
        st.markdown("---")
        st.subheader("📈 Prediction Visualization")
        
        # Get train and test data
        train_data, test_data = processor.get_train_test_data()
        test_data = test_data.copy()
        test_data['Predictions'] = predictions
        
        # Create visualizer
        viz = Visualizer()
        
        # Plot predictions
        fig = viz.plot_lstm_predictions(train_data, test_data, predictions)
        st.pyplot(fig)
        
        # Training history
        st.markdown("---")
        st.subheader("📉 Training History")
        fig_history = viz.plot_training_history(history)
        st.pyplot(fig_history)
        
        # Save model
        if st.button("💾 Save Trained Model"):
            lstm_model.save_model()
            st.success("✅ Model saved successfully!")
        
        # Store in session state for comparison tab
        st.session_state['lstm_predictions'] = predictions
        st.session_state['lstm_metrics'] = metrics
        st.session_state['test_data'] = test_data
        st.session_state['train_data'] = train_data


# ===========================
# TAB 2: XGBOOST MODEL
# ===========================
with tab2:
    st.header("🟣 XGBoost Model - Gradient Boosted Trees")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### About XGBoost Model
        XGBoost (eXtreme Gradient Boosting) is a powerful machine learning algorithm that uses 
        gradient boosting on decision trees. Unlike LSTM, it doesn't have built-in memory, 
        so we engineer features to capture temporal patterns.
        
        **Key Features:**
        - Uses 13 engineered features (Moving Averages, RSI, Lag prices, etc.)
        - Fast training compared to deep learning
        - Handles non-linear relationships well
        - Feature importance visualization
        """)
    
    with col2:
        st.markdown("""
        ### Engineered Features
        - **Moving Averages**: 7, 21, 50 days
        - **RSI**: Momentum indicator
        - **Time Features**: Day, Month, Quarter
        - **Lag Features**: Previous 3 days prices
        - **Volatility**: Rolling std deviation
        - **Price Changes**: Absolute & percentage
        """)
    
    st.markdown("---")
    
    # Train button
    if st.button("🚀 Train XGBoost Model", type="primary", use_container_width=True):
        
        with st.spinner("🔄 Creating features..."):
            # Create XGBoost features
            X_train, X_test, y_train, y_test = processor.create_xgboost_features()
        
        st.success("✅ Features created successfully!")
        
        # Display feature info
        feature_names = [
            'MA_Ratio_7_21', 'MA_Ratio_21_50',
            'RSI', 
            'Day', 'Month',
            'Lag_1', 'Lag_2', 'Lag_3', 'Lag_4', 'Lag_5',
            'Price_Change_Pct_1d', 'Price_Change_Pct_5d',
            'Volatility_10', 'Volatility_30'
        ]
        
        with st.expander("📊 View Feature Statistics"):
           
            feature_df = pd.DataFrame(X_train, columns=feature_names)
            st.dataframe(feature_df.describe(), use_container_width=True)
        
        # Initialize model
       
        xgb_model = XGBoostModel(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate
        )
        
        # Training
        with st.spinner(f"🤖 Training XGBoost model... This is faster than LSTM!"):
            progress_bar = st.progress(0)
            
            xgb_model.train(X_train, y_train, verbose=False)
            
            progress_bar.progress(100)
        
        st.success("✅ Training completed!")
        
        # Make predictions
        with st.spinner("🔮 Making predictions..."):
            predictions = xgb_model.predict(X_test)
        
        # Calculate metrics
        metrics = get_metrics(y_test, predictions)
        
        # Display metrics
        st.subheader("📊 Model Performance")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                label="Root Mean Squared Error (RMSE)",
                value=f"{metrics['RMSE']:.4f}",
                delta=None
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                label="Mean Absolute Error (MAE)",
                value=f"{metrics['MAE']:.4f}",
                delta=None
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.info(f"""
        **Interpretation:**
        - RMSE of {metrics['RMSE']:.4f} means predictions are off by approximately {metrics['RMSE']:.4f} on average.
        - MAE of {metrics['MAE']:.4f} shows the average absolute difference between predicted and actual prices.
        """)
        
        # Visualizations
        st.markdown("---")
        st.subheader("📈 Prediction Visualization")
        
        # Get train and test data for XGBoost
        train_data_xgb, test_data_xgb = processor.get_xgboost_train_test_data()
        test_data_xgb = test_data_xgb.copy()
        test_data_xgb['Predictions'] = predictions
        
        # Create visualizer
        viz = Visualizer()
        
        # Plot predictions (reuse LSTM plotting function)
        fig = viz.plot_lstm_predictions(
            train_data_xgb, 
            test_data_xgb, 
            predictions,
            title="XGBoost Stock Price Prediction"
        )
        st.pyplot(fig)
        
        # Feature Importance
        st.markdown("---")
        st.subheader("🎯 Feature Importance")
        st.markdown("See which features contributed most to the predictions:")
        
        feature_importance = xgb_model.get_feature_importance()
        fig_importance = viz.plot_feature_importance(feature_importance, feature_names)
        st.pyplot(fig_importance)
        
        # Save model
        if st.button("💾 Save Trained XGBoost Model"):
            xgb_model.save_model()
            st.success("✅ Model saved successfully!")
        
        # Store in session state for comparison tab
        st.session_state['xgb_predictions'] = predictions
        st.session_state['xgb_metrics'] = metrics
        st.session_state['xgb_test_data'] = test_data_xgb
        st.session_state['xgb_train_data'] = train_data_xgb

# ===========================
# TAB 3: COMPARISON
# ===========================
with tab3:
    st.header("⚖️ Model Comparison - LSTM vs XGBoost")
    
    # Check if both models are trained
    lstm_trained = 'lstm_predictions' in st.session_state
    xgb_trained = 'xgb_predictions' in st.session_state
    
    if lstm_trained and xgb_trained:
        
        # Performance Metrics Comparison
        st.subheader("📊 Performance Metrics Comparison")
        
        comparison_df = pd.DataFrame({
            'Model': ['LSTM', 'XGBoost'],
            'RMSE': [
                f"{st.session_state['lstm_metrics']['RMSE']:.4f}",
                f"{st.session_state['xgb_metrics']['RMSE']:.4f}"
            ],
            'MAE': [
                f"{st.session_state['lstm_metrics']['MAE']:.4f}",
                f"{st.session_state['xgb_metrics']['MAE']:.4f}"
            ],
            'Status': ['✅ Trained', '✅ Trained']
        })
        
        st.dataframe(comparison_df, use_container_width=True)
        
        # Determine winner
        lstm_rmse = st.session_state['lstm_metrics']['RMSE']
        xgb_rmse = st.session_state['xgb_metrics']['RMSE']
        
        if lstm_rmse < xgb_rmse:
            winner = "🏆 LSTM performs better!"
            diff = xgb_rmse - lstm_rmse
            st.success(f"{winner} LSTM has {diff:.4f} lower RMSE than XGBoost.")
        else:
            winner = "🏆 XGBoost performs better!"
            diff = lstm_rmse - xgb_rmse
            st.success(f"{winner} XGBoost has {diff:.4f} lower RMSE than LSTM.")
        
        # Side-by-side comparison visualization
        st.markdown("---")
        st.subheader("📈 Visual Comparison")
        
        viz = Visualizer()
        
        # Use LSTM's train/test data (they should have similar date ranges)
        fig_comparison = viz.plot_comparison(
            st.session_state['train_data'],
            st.session_state['test_data'],
            st.session_state['lstm_predictions'],
            st.session_state['xgb_predictions']
        )
        st.pyplot(fig_comparison)
        
        # Analysis
        st.markdown("---")
        st.subheader("🔍 Analysis: Why Different Results?")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### LSTM Approach
            - **Sequential Learning**: Learns patterns from 60-day sequences
            - **Memory**: Has built-in memory to remember past trends
            - **Deep Learning**: Multiple layers capture complex patterns
            - **Best for**: Long-term dependencies and trends
            """)
        
        with col2:
            st.markdown("""
            ### XGBoost Approach
            - **Feature-Based**: Uses engineered features (MA, RSI, etc.)
            - **Tree Ensemble**: Combines multiple decision trees
            - **Fast Training**: Much quicker than neural networks
            - **Best for**: Non-linear relationships and interpretability
            """)
        
        st.info("""
        **Key Differences:**
        - LSTM sees the data as a sequence of prices over time
        - XGBoost sees each day as independent features (moving averages, indicators, etc.)
        - LSTM might capture long-term trends better
        - XGBoost might react faster to recent changes (via lag features)
        - Performance depends on the specific stock and time period
        """)
        
    elif lstm_trained and not xgb_trained:
        st.warning("⚠️ Please train the XGBoost model to see full comparison!")
        st.info("Go to the **XGBoost Model** tab and click 'Train XGBoost Model'")
        
    elif not lstm_trained and xgb_trained:
        st.warning("⚠️ Please train the LSTM model to see full comparison!")
        st.info("Go to the **LSTM Model** tab and click 'Train LSTM Model'")
        
    else:
        st.warning("⚠️ Please train both models to see comparison!")
        st.info("Train LSTM in tab 1 and XGBoost in tab 2, then come back here.")