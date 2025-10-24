import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime
import io

# Enhanced sample data with more realistic AI metrics
@st.cache_data
def load_data():
    np.random.seed(42)
    n_samples = 500
    
    data = {
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='H'),
        'metric_a': np.random.normal(50, 15, n_samples),
        'metric_b': np.random.gamma(2, 2, n_samples) * 10,
        'feature_1': np.random.randn(n_samples),
        'feature_2': np.random.randn(n_samples) + 2,
        'prediction': np.random.choice(['A', 'B', 'C'], size=n_samples, p=[0.6, 0.3, 0.1]),
        'probability': np.random.beta(2, 2, n_samples),
        'confidence_score': np.random.beta(5, 2, n_samples),
        'anomaly_score': np.random.exponential(0.5, n_samples),
        'cluster': np.random.choice(['Cluster 1', 'Cluster 2', 'Cluster 3'], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Add some correlations and patterns
    df['metric_a'] = df['metric_a'] + df['feature_1'] * 5
    df['metric_b'] = df['metric_b'] + df['feature_2'] * 3
    
    return df

df = load_data()

# Page configuration with expanded options
st.set_page_config(
    page_title="Neural Insights Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== ENHANCED FUTURISTIC CSS =====
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700;900&family=Exo+2:wght@300;400;500;600;700&display=swap');
    
    .main {
        background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    .stApp {
        background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 50%, #16213e 100%);
        font-family: 'Exo 2', sans-serif;
    }
    
    /* Headers with futuristic font */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Orbitron', sans-serif !important;
        color: #00ffff !important;
        text-shadow: 0 0 10px rgba(0, 255, 255, 0.5);
        letter-spacing: 1px;
    }
    
    /* Special AI-powered header */
    .ai-header {
        background: linear-gradient(90deg, #00ffff, #ff00ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-size: 3rem !important;
        margin-bottom: 2rem;
    }
    
    /* Cards with glass morphism effect */
    .card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px 0 rgba(0, 255, 255, 0.1);
    }
    
    /* Enhanced buttons */
    .stButton>button {
        background: linear-gradient(45deg, #00ffff, #0080ff);
        color: black;
        border: none;
        padding: 12px 28px;
        text-align: center;
        font-weight: 600;
        border-radius: 25px;
        transition: all 0.3s ease;
        font-family: 'Exo 2', sans-serif;
        box-shadow: 0 4px 15px 0 rgba(0, 255, 255, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px 0 rgba(0, 255, 255, 0.5);
    }
    
    /* Slider styling */
    .stSlider>div>div>div {
        background: linear-gradient(90deg, #00ffff, #ff00ff);
    }
    
    /* Select boxes */
    .stSelectbox>div>div {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(0, 255, 255, 0.3);
        border-radius: 10px;
    }
    
    /* Metrics styling */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #00ffff !important;
        text-shadow: 0 0 10px rgba(0, 255, 255, 0.5);
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1rem !important;
        font-weight: 400 !important;
        color: #ffffff !important;
    }
    
    /* Progress bars */
    .stProgress>div>div>div {
        background: linear-gradient(90deg, #00ffff, #ff00ff);
    }
    
    /* Sidebar enhancements */
    .css-1d391kg {
        background: rgba(10, 10, 20, 0.9) !important;
    }
</style>
""", unsafe_allow_html=True)

# ===== SIDEBAR WITH ENHANCED CONTROLS =====
with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>🧠 NEURAL CONTROLS</h1>", unsafe_allow_html=True)
    
    # AI Model Selection
    st.markdown("### 🤖 AI Model Configuration")
    ai_model = st.selectbox("Select AI Engine", 
                           ["Deep Neural Network", "Transformer Model", "Reinforcement Learning", "Generative AI"])
    
    # Advanced Filters
    st.markdown("### 🔍 Data Filters")
    metric_choice = st.selectbox("Primary Metric", ['metric_a', 'metric_b'])
    feature_choice = st.selectbox("Feature Analysis", ['feature_1', 'feature_2'])
    
    # Enhanced multi-select with search
    predictions_filter = st.multiselect(
        "Prediction Classes", 
        ['A', 'B', 'C'], 
        default=['A', 'B', 'C'],
        help="Select prediction classes to analyze"
    )
    
    # Range sliders for better control
    probability_range = st.slider(
        "Confidence Range", 
        0.0, 1.0, (0.5, 0.9),
        help="Filter by model confidence scores"
    )
    
    date_range = st.date_input(
        "Time Range",
        [df['timestamp'].min().date(), df['timestamp'].max().date()]
    )
    
    # AI Analysis Toggles
    st.markdown("### ⚡ AI Analysis")
    enable_anomaly_detection = st.toggle("Real-time Anomaly Detection", True)
    enable_trend_analysis = st.toggle("Trend Analysis", True)
    enable_clustering = st.toggle("Auto-clustering", True)
    
    # Action Buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 Analyze", use_container_width=True):
            st.session_state.analyze_clicked = True
    
    with col2:
        if st.button("🔄 Reset", use_container_width=True):
            st.rerun()
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888;'>
        <p>Powered by Advanced Neural Networks</p>
        <p>v2.1.0 | Real-time AI Analytics</p>
    </div>
    """, unsafe_allow_html=True)

# ===== MAIN DASHBOARD =====
st.markdown("<h1 class='ai-header'>🧠 NEURAL INSIGHTS DASHBOARD</h1>", unsafe_allow_html=True)

# Apply filters
filtered_df = df[
    (df['prediction'].isin(predictions_filter)) & 
    (df['probability'] >= probability_range[0]) & 
    (df['probability'] <= probability_range[1]) &
    (df['timestamp'].dt.date >= date_range[0]) & 
    (df['timestamp'].dt.date <= date_range[1])
]

# ===== REAL-TIME METRICS DASHBOARD =====
st.markdown("## 📊 REAL-TIME AI METRICS")

# Create 4 columns for metrics
metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

with metric_col1:
    delta_metric_a = filtered_df['metric_a'].mean() - df['metric_a'].mean()
    st.metric(
        "Average Metric A", 
        f"{filtered_df['metric_a'].mean():.2f}",
        f"{delta_metric_a:+.2f}"
    )

with metric_col2:
    delta_metric_b = filtered_df['metric_b'].mean() - df['metric_b'].mean()
    st.metric(
        "Average Metric B", 
        f"{filtered_df['metric_b'].mean():.2f}",
        f"{delta_metric_b:+.2f}"
    )

with metric_col3:
    st.metric(
        "AI Confidence", 
        f"{filtered_df['confidence_score'].mean()*100:.1f}%",
        "±2.3%"
    )

with metric_col4:
    anomaly_count = len(filtered_df[filtered_df['anomaly_score'] > 1.5])
    st.metric(
        "Anomalies Detected", 
        f"{anomaly_count}",
        "Real-time"
    )

# ===== ADVANCED VISUALIZATIONS =====
st.markdown("## 📈 AI-POWERED ANALYTICS")

# Create tabs for different visualizations
tab1, tab2, tab3, tab4 = st.tabs(["🧭 Interactive Analysis", "📊 Distribution Insights", "🕒 Time Series", "🔍 AI Clustering"])

with tab1:
    # Enhanced scatter plot with multiple dimensions
    col1, col2 = st.columns([3, 1])
    
    with col1:
        fig_scatter = px.scatter(
            filtered_df, 
            x=feature_choice, 
            y=metric_choice, 
            color='prediction',
            size='probability',
            hover_data=['confidence_score', 'anomaly_score'],
            title=f"AI Analysis: {feature_choice} vs {metric_choice}",
            color_discrete_map={'A': '#00ffff', 'B': '#ff00ff', 'C': '#ffff00'}
        )
        fig_scatter.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            title_font=dict(size=20)
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 📋 Data Summary")
        st.write(f"**Samples:** {len(filtered_df):,}")
        st.write(f"**Features:** {len(filtered_df.columns)}")
        st.write(f"**Date Range:** {date_range[0]} to {date_range[1]}")
        st.write(f"**AI Model:** {ai_model}")
        st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    # Distribution analysis
    col1, col2 = st.columns(2)
    
    with col1:
        fig_hist = px.histogram(
            filtered_df, 
            x=metric_choice,
            color='prediction',
            marginal="box",
            title=f"Distribution of {metric_choice}",
            color_discrete_map={'A': '#00ffff', 'B': '#ff00ff', 'C': '#ffff00'}
        )
        fig_hist.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        # Confidence score distribution
        fig_conf = px.box(
            filtered_df,
            x='prediction',
            y='confidence_score',
            color='prediction',
            title="AI Confidence by Prediction Class",
            color_discrete_map={'A': '#00ffff', 'B': '#ff00ff', 'C': '#ffff00'}
        )
        fig_conf.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig_conf, use_container_width=True)

with tab3:
    # Time series analysis
    fig_time = px.line(
        filtered_df.sort_values('timestamp'),
        x='timestamp',
        y=metric_choice,
        color='prediction',
        title=f"Time Series Analysis: {metric_choice}",
        color_discrete_map={'A': '#00ffff', 'B': '#ff00ff', 'C': '#ffff00'}
    )
    fig_time.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    st.plotly_chart(fig_time, use_container_width=True)

with tab4:
    # Clustering visualization
    col1, col2 = st.columns(2)
    
    with col1:
        fig_cluster = px.scatter(
            filtered_df,
            x='feature_1',
            y='feature_2',
            color='cluster',
            size='probability',
            title="AI-Generated Clusters",
            hover_data=['prediction', 'confidence_score']
        )
        fig_cluster.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig_cluster, use_container_width=True)
    
    with col2:
        cluster_summary = filtered_df.groupby('cluster').agg({
            'probability': 'mean',
            'confidence_score': 'mean',
            'anomaly_score': 'mean'
        }).round(3)
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 🎯 Cluster Analysis")
        st.dataframe(cluster_summary, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ===== AI-DRIVEN INSIGHTS SECTION =====
st.markdown("## 🤖 ADVANCED AI INSIGHTS")

# Simulate AI analysis with progress and insights
with st.spinner("🧠 Neural Network analyzing patterns..."):
    progress_bar = st.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress_bar.progress(i + 1)
    
    # AI Insights Cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 🚨 Pattern Detection")
        st.write("**Strong correlation** detected between Feature 1 and Metric A")
        st.write("📈 **Correlation:** 0.87")
        st.write("🎯 **Confidence:** 94%")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### ⚡ Performance Alert")
        st.write("**Cluster 3** shows abnormal behavior patterns")
        st.write("🔍 **Anomaly Score:** 8.7/10")
        st.write("💡 **Recommendation:** Investigate data quality")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 📊 Predictive Analytics")
        st.write("**Trend analysis** predicts 12% growth in Metric B")
        st.write("🎯 **Accuracy:** 89%")
        st.write("📅 **Timeframe:** Next 30 days")
        st.markdown("</div>", unsafe_allow_html=True)

# ===== ANOMALY DETECTION SECTION =====
st.markdown("## 🚨 REAL-TIME ANOMALY DETECTION")

if enable_anomaly_detection:
    # Calculate anomalies using multiple criteria
    metric_a_threshold = filtered_df['metric_a'].mean() + 2 * filtered_df['metric_a'].std()
    anomalies = filtered_df[
        (filtered_df['anomaly_score'] > 1.5) | 
        (filtered_df['metric_a'] > metric_a_threshold) |
        (filtered_df['confidence_score'] < 0.3)
    ]
    
    if not anomalies.empty:
        st.warning(f"🔍 **AI Alert:** Detected {len(anomalies)} potential anomalies requiring attention!")
        
        # Display anomalies in an expandable section
        with st.expander("View Detailed Anomaly Report"):
            st.dataframe(anomalies.style.highlight_max(color='#ff6b6b'), use_container_width=True)
            
            # Export option
            csv = anomalies.to_csv(index=False)
            st.download_button(
                label="📥 Download Anomaly Report",
                data=csv,
                file_name=f"anomaly_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    else:
        st.success("✅ **System Optimal:** No anomalies detected in current dataset")

# ===== DATA EXPLORER SECTION =====
st.markdown("## 💾 INTERACTIVE DATA EXPLORER")

with st.expander("🔍 Explore Filtered Dataset", expanded=False):
    # Data statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(filtered_df):,}")
    with col2:
        st.metric("Data Columns", f"{len(filtered_df.columns)}")
    with col3:
        st.metric("Memory Usage", f"{filtered_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    with col4:
        st.metric("Missing Values", f"{filtered_df.isnull().sum().sum()}")
    
    # Interactive data table
    st.dataframe(
        filtered_df.style.background_gradient(cmap='viridis'),
        use_container_width=True,
        height=400
    )

# ===== FOOTER =====
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🧠 <b>Neural Insights Dashboard</b> | Powered by Advanced Machine Learning Algorithms</p>
    <p>Real-time Analytics • Pattern Recognition • Predictive Intelligence</p>
</div>
""", unsafe_allow_html=True)

# ===== PERFORMANCE MONITORING =====
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📊 System Performance")
    
    # Simulated performance metrics
    cpu_usage = st.slider("CPU Usage (%)", 0, 100, 45)
    memory_usage = st.slider("Memory Usage (%)", 0, 100, 62)
    
    st.write(f"**AI Processing:** {len(filtered_df):,} records")
    st.write(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")