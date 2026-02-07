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
    epochs = st.slider("Training Epochs", min_value=10, max_value=50, value=20, step=5)
    batch_size = st.slider("Batch Size", min_value=16, max_value=64, value=32, step=16)
    
    st.markdown("---")
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
    st.header("🟣 XGBoost Model")
    st.info("⏳ XGBoost model implementation coming soon! (Tomorrow's work)")
    
    st.markdown("""
    ### What is XGBoost?
    XGBoost (eXtreme Gradient Boosting) is a powerful machine learning algorithm that uses 
    decision trees and gradient boosting.
    
    **For stock prediction, we'll use:**
    - Moving Averages (7, 21, 50 days)
    - RSI (Relative Strength Index)
    - Day of week and month features
    - Lagged price features
    """)

# ===========================
# TAB 3: COMPARISON
# ===========================
with tab3:
    st.header("⚖️ Model Comparison")
    
    if 'lstm_predictions' in st.session_state:
        st.subheader("📊 Performance Metrics Comparison")
        
        # Create comparison table
        comparison_df = pd.DataFrame({
            'Model': ['LSTM', 'XGBoost'],
            'RMSE': [st.session_state['lstm_metrics']['RMSE'], 'Coming Soon'],
            'MAE': [st.session_state['lstm_metrics']['MAE'], 'Coming Soon'],
            'Status': ['✅ Trained', '⏳ Pending']
        })
        
        st.dataframe(comparison_df, use_container_width=True)
        
        st.info("Full comparison will be available once both models are trained!")
    else:
        st.warning("⚠️ Please train the LSTM model first to see comparison!")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>📚 BCA Final Year Project | Stock Price Prediction System</p>
        <p>Built with Streamlit, TensorFlow, and XGBoost</p>
    </div>
""", unsafe_allow_html=True)