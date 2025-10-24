import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px # Use plotly for interactive plots
import time  # for adding loading animations

# Sample DataFrame (Replace with your actual data)
@st.cache_data  #Cache the data to improve performance
def load_data():
    np.random.seed(42)
    data = {
        'metric_a': np.random.rand(100),
        'metric_b': np.random.rand(100) * 10,
        'feature_1': np.random.randn(100),
        'feature_2': np.random.randn(100) + 2,
        'prediction': np.random.choice(['A', 'B', 'C'], size=100),
        'probability': np.random.rand(100)
    }
    return pd.DataFrame(data)

df = load_data()

# 1. Streamlit App Layout and Styling (Futuristic Theme)
st.set_page_config(page_title="AI-Powered Dashboard", page_icon=":rocket:", layout="wide")  # Full width

# --- CSS for Futuristic Theme ---
st.markdown(
    """
    <style>
    body {
        background-color: #0E1117;  /* Dark background */
        color: #FAFAFA;             /* Light text */
    }
    .stApp {
        background-color: #0E1117;
    }
    .stButton>button {
        color: #FAFAFA;
        background-color: #4CAF50; /* Green button */
        border: none;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 5px;
    }
    .stSlider>div[data-baseweb="slider"] > div[data-testid="stThumb"] {
        background-color: #4CAF50;
    }
    .stSelectbox > div > button {
        color: #FAFAFA;
    }
    .stTextInput>label {
        color: #FAFAFA;
    }
    .stNumberInput>label {
        color: #FAFAFA;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #FAFAFA;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Sidebar ---
with st.sidebar:
    st.title("Dashboard Controls")
    st.markdown("Explore your data with these interactive filters.")

    metric_choice = st.selectbox("Select Metric", ['metric_a', 'metric_b'])
    feature_choice = st.selectbox("Select Feature", ['feature_1', 'feature_2'])
    predictions_filter = st.multiselect("Select Prediction(s)", ['A', 'B', 'C'], default=['A', 'B', 'C']) # Select multiple predictions
    probability_threshold = st.slider("Probability Threshold", 0.0, 1.0, 0.5, 0.01) # Slider for threshold

    # Add a "reset" button to clear all filters
    if st.button("Reset Filters"):
        predictions_filter = ['A', 'B', 'C'] # Set defaults
        probability_threshold = 0.5  # Set default
        st.rerun() # Rerun the app to reset state

    st.markdown("---")
    st.write("Made with ❤️ by [Your Name/Team]") # Add branding
    st.markdown("[Source Code](link_to_your_repo)")

# 2. Main Content
st.title("Interactive Data Explorer")

# --- Data Filtering ---
filtered_df = df[df['prediction'].isin(predictions_filter)]
filtered_df = filtered_df[filtered_df['probability'] >= probability_threshold]

# --- Metrics Display ---
st.header("Key Metrics")
col1, col2 = st.columns(2) # two columns to display two metrics side-by-side

with col1:
  avg_metric_a = filtered_df['metric_a'].mean()
  st.metric("Average Metric A", f"{avg_metric_a:.2f}")

with col2:
  avg_metric_b = filtered_df['metric_b'].mean()
  st.metric("Average Metric B", f"{avg_metric_b:.2f}")

# --- Data Visualization (Plotly) ---
st.header("Data Visualization")
fig = px.scatter(filtered_df, x=feature_choice, y=metric_choice, color='prediction', size='probability', hover_data=filtered_df.columns) #more interactive plot
st.plotly_chart(fig, use_container_width=True)

# --- Data Table ---
st.header("Filtered Data")
st.dataframe(filtered_df) #Show the filtered data

# --- AI-Driven Insight (Example - Simple Anomaly Detection) ---
st.header("AI-Driven Insights")

# Simple example:  Identify data points far from the mean of metric_a
metric_a_mean = df['metric_a'].mean()
metric_a_std = df['metric_a'].std()
anomaly_threshold = 2  # Adjust this threshold as needed
anomalies = df[(df['metric_a'] > metric_a_mean + anomaly_threshold * metric_a_std) | (df['metric_a'] < metric_a_mean - anomaly_threshold * metric_a_std)]

if not anomalies.empty:
    st.warning(f"Detected {len(anomalies)} potential anomalies based on Metric A.")
    st.dataframe(anomalies)
else:
    st.success("No anomalies detected based on Metric A.")

# --- Loading Animation Example ---
with st.spinner("Analyzing data..."):
    time.sleep(2)  # Simulate a long computation
st.success("Analysis complete!")
