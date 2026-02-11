"""
Streamlit Dashboard for Adaptive Demand Forecasting
Features: Forecasts, Business Rules, Drift Diagnostics, Model Training & Improvement
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
import yaml
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_ingestion import load_pos_data, load_external_data, merge_data, load_stock_data
from src.feature_engineering import prepare_features, get_feature_columns
from src.forecasting import ForecastingEngine
from src.business_rules import BusinessRuleEngine, apply_business_rules
from src.drift_detection import DriftMonitor, check_drift
from src.diagnostics import generate_drift_report, DiagnosticsEngine

# Page config
st.set_page_config(
    page_title="Demand Forecasting Dashboard",
    page_icon="📊",
    layout="wide"
)

# Initialize paths
BASE_PATH = Path(__file__).parent.parent
DATA_PATH = BASE_PATH / "data"
MODELS_PATH = BASE_PATH / "models"
CONFIG_PATH = BASE_PATH / "config"
LOGS_PATH = BASE_PATH / "logs"


@st.cache_data
def load_data():
    """Load and prepare all data."""
    pos_df = load_pos_data(DATA_PATH / "pos_data.csv")
    external_df = load_external_data(DATA_PATH / "external_data.csv")
    merged = merge_data(pos_df, external_df)
    features = prepare_features(merged)
    return features


@st.cache_resource
def get_forecasting_engine():
    """Initialize forecasting engine."""
    engine = ForecastingEngine(str(MODELS_PATH))
    try:
        engine.load_all_models()
    except:
        pass
    return engine


def load_drift_events():
    """Load drift events from log file."""
    log_file = LOGS_PATH / "model_events.csv"
    if log_file.exists():
        df = pd.read_csv(log_file)
        if len(df) > 0 and 'event_type' in df.columns:
            return df[df['event_type'] == 'DRIFT_DETECTED']
    return pd.DataFrame(columns=['timestamp', 'event_type', 'sku', 'details'])


def save_override(sku: str, override_qty: int, reason: str):
    """Save user override to CSV."""
    override_file = DATA_PATH / "overrides.csv"
    new_row = pd.DataFrame([{
        'sku': sku,
        'date': datetime.now().strftime('%Y-%m-%d'),
        'override_qty': override_qty,
        'reason': reason,
        'user': 'dashboard_user',
        'timestamp': datetime.now().isoformat()
    }])
    
    if override_file.exists():
        existing = pd.read_csv(override_file)
        updated = pd.concat([existing, new_row], ignore_index=True)
    else:
        updated = new_row
    
    updated.to_csv(override_file, index=False)


# Main app
st.title("📊 Adaptive Demand Forecasting Dashboard")
st.markdown("---")

# Load data
try:
    df = load_data()
    engine = get_forecasting_engine()
    skus = df['sku'].unique().tolist()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Select Page", [
    "📈 Forecast & Orders", 
    "🎯 Model Training", 
    "⚙️ Business Rules", 
    "🔍 Drift & Diagnostics"
])
st.sidebar.markdown("---")
st.sidebar.info(f"📊 Data: {len(df)} records\n🏷️ SKUs: {len(skus)}\n🤖 Models: {len(engine.models)}")

# Page 1: Forecast & Orders
if page == "📈 Forecast & Orders":
    st.header("📈 Forecast & Orders")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        selected_sku = st.selectbox("Select SKU", skus)
        horizon = st.slider("Forecast Horizon (days)", 3, 14, 7)
        
        # Show model info
        if selected_sku in engine.models:
            model = engine.models[selected_sku]
            st.metric("Model Version", f"v{model.version}")
            if model.metrics:
                st.metric("Current MAE", f"{model.metrics.get('mae', 0):.1f}")
    
    # Get forecast data
    sku_df = df[df['sku'] == selected_sku].copy()
    
    # Train model if needed
    if selected_sku not in engine.models:
        with st.spinner(f"Training model for {selected_sku}..."):
            from src.forecasting import ForecastModel
            model = ForecastModel(selected_sku)
            model.train(sku_df, get_feature_columns())
            engine.models[selected_sku] = model
    
    # Generate predictions
    model = engine.models[selected_sku]
    predictions = model.predict(sku_df.tail(horizon))
    
    # Plot historical + forecast
    with col2:
        fig = go.Figure()
        
        # Historical (last 60 days for clarity)
        recent_df = sku_df.tail(60)
        fig.add_trace(go.Scatter(
            x=recent_df['date'],
            y=recent_df['units_sold'],
            mode='lines+markers',
            name='Historical Sales',
            line=dict(color='#2E86AB', width=2),
            marker=dict(size=4)
        ))
        
        # Forecast
        forecast_dates = sku_df['date'].tail(horizon)
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=predictions,
            mode='lines+markers',
            name=f'Forecast (v{model.version})',
            line=dict(color='#F18F01', dash='dash', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title=f"Demand Forecast: {selected_sku}",
            xaxis_title="Date",
            yaxis_title="Units",
            height=400,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Orders table
    st.subheader("Order Recommendations")
    
    stock_df = load_stock_data()
    rules_engine = BusinessRuleEngine(str(CONFIG_PATH / "business_rules.yml"))
    
    # Calculate orders for all SKUs
    order_data = []
    for sku in skus:
        sku_data = df[df['sku'] == sku]
        if sku in engine.models:
            preds = engine.models[sku].predict(sku_data.tail(horizon))
            total_forecast = preds.sum()
        else:
            total_forecast = sku_data['units_sold'].tail(horizon).sum()
        
        stock = stock_df[stock_df['sku'] == sku]['current_stock'].values
        current_stock = stock[0] if len(stock) > 0 else 0
        
        result = rules_engine.apply_rules(
            forecast_qty=total_forecast,
            current_stock=current_stock,
            sku=sku,
            daily_demand=total_forecast / horizon
        )
        
        order_data.append({
            'SKU': sku,
            'Forecast (7d)': int(total_forecast),
            'Current Stock': int(current_stock),
            'Order Qty': result['final_qty'],
            'Constraint': result['explanation']
        })
    
    order_df = pd.DataFrame(order_data)
    st.dataframe(order_df, use_container_width=True)
    
    # Override section
    with st.expander("📝 Manual Override"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            override_sku = st.selectbox("Override SKU", skus, key="override_sku")
        with col2:
            override_qty = st.number_input("Override Quantity", min_value=0, value=0)
        with col3:
            override_reason = st.text_input("Reason for Override")
        
        if st.button("Submit Override"):
            if override_qty > 0 and override_reason:
                save_override(override_sku, override_qty, override_reason)
                st.success(f"Override saved for {override_sku}: {override_qty} units")
            else:
                st.warning("Please enter quantity and reason")

# Page 2: Model Training
elif page == "🎯 Model Training":
    st.header("🎯 Model Training & Improvement")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Train Models")
        
        # Training options
        train_mode = st.radio(
            "Training Mode",
            ["Train All SKUs", "Train Single SKU"],
            help="Choose to train all models or a specific one"
        )
        
        if train_mode == "Train Single SKU":
            train_sku = st.selectbox("Select SKU to Train", skus)
        
        # Date range for training
        st.markdown("**Training Data Range**")
        date_col1, date_col2 = st.columns(2)
        min_date = df['date'].min().date()
        max_date = df['date'].max().date()
        
        with date_col1:
            start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
        with date_col2:
            end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
        
        if st.button("🚀 Start Training", type="primary"):
            train_df = df[(df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date)]
            
            if len(train_df) < 20:
                st.error("Insufficient training data. Select a larger date range.")
            else:
                with st.spinner("Training models..."):
                    if train_mode == "Train All SKUs":
                        results = engine.train_all(train_df, get_feature_columns())
                        st.success(f"✅ Trained {len(results)} models!")
                        
                        # Show results
                        result_df = pd.DataFrame([
                            {'SKU': sku, 'MAE': m['mae'], 'R²': m['r2'], 'Version': m['version']}
                            for sku, m in results.items()
                        ])
                        st.dataframe(result_df.round(3), use_container_width=True)
                    else:
                        metrics = engine.retrain_sku(train_sku, train_df, get_feature_columns())
                        st.success(f"✅ Trained {train_sku} v{metrics['version']}")
                        st.metric("MAE", f"{metrics['mae']:.2f}")
                        st.metric("R²", f"{metrics['r2']:.3f}")
                
                # Clear cache to reload models
                st.cache_resource.clear()
    
    with col2:
        st.subheader("📈 Model Improvement")
        
        # Show improvement report
        if len(engine.models) > 0:
            try:
                report = engine.get_improvement_report()
                
                if len(report) > 0 and 'improvement_pct' in report.columns:
                    # Create improvement chart
                    fig = go.Figure()
                    
                    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in report['improvement_pct']]
                    
                    fig.add_trace(go.Bar(
                        x=report['sku'],
                        y=report['improvement_pct'],
                        marker_color=colors,
                        text=[f"{x:+.1f}%" for x in report['improvement_pct']],
                        textposition='outside'
                    ))
                    
                    fig.update_layout(
                        title="Model Improvement (MAE Reduction %)",
                        xaxis_title="SKU",
                        yaxis_title="Improvement %",
                        height=300,
                        yaxis=dict(zeroline=True, zerolinecolor='gray')
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Improvement table
                    st.dataframe(
                        report[['sku', 'versions', 'initial_mae', 'current_mae', 'improvement_pct']].round(2),
                        use_container_width=True
                    )
                else:
                    st.info("Train models multiple times to see improvement tracking")
            except Exception as e:
                st.warning(f"Improvement tracking not available: {e}")
        else:
            st.info("No models trained yet. Train models to see improvement tracking.")
        
        # Model versions
        st.subheader("🏷️ Model Versions")
        if len(engine.models) > 0:
            version_data = []
            for sku, model in engine.models.items():
                version_data.append({
                    'SKU': sku,
                    'Version': f"v{model.version}",
                    'MAE': model.metrics.get('mae', '-'),
                    'R²': model.metrics.get('r2', '-'),
                    'Samples': model.metrics.get('train_samples', 0) + model.metrics.get('val_samples', 0)
                })
            
            version_df = pd.DataFrame(version_data)
            st.dataframe(version_df, use_container_width=True)

# Page 3: Business Rules
elif page == "⚙️ Business Rules":
    st.header("⚙️ Business Rules Configuration")
    
    rules_engine = BusinessRuleEngine(str(CONFIG_PATH / "business_rules.yml"))
    rules_df = rules_engine.get_rules_df()
    
    st.markdown("Edit the rules below and click Save to update the configuration.")
    
    # Editable dataframe
    edited_df = st.data_editor(
        rules_df,
        column_config={
            "sku": st.column_config.TextColumn("SKU", disabled=True),
            "max_shelf_capacity": st.column_config.NumberColumn("Max Shelf Capacity", min_value=1),
            "unit_cost": st.column_config.NumberColumn("Unit Cost ($)", min_value=0.01, format="%.2f"),
            "max_budget_per_order": st.column_config.NumberColumn("Max Budget ($)", min_value=1),
            "perishability_days": st.column_config.NumberColumn("Shelf Life (days)", min_value=1),
            "safety_stock_days": st.column_config.NumberColumn("Safety Stock (days)", min_value=0)
        },
        use_container_width=True,
        num_rows="fixed"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("💾 Save Rules", type="primary"):
            for _, row in edited_df.iterrows():
                sku = row['sku']
                for col in ['max_shelf_capacity', 'unit_cost', 'max_budget_per_order', 
                           'perishability_days', 'safety_stock_days']:
                    rules_engine.update_rule(sku, col, row[col])
            
            rules_engine.save_rules(str(CONFIG_PATH / "business_rules.yml"))
            st.success("Rules saved successfully!")
            st.cache_data.clear()
    
    # Rules explanation
    st.markdown("---")
    st.subheader("📖 Rule Descriptions")
    st.markdown("""
    | Rule | Description |
    |------|-------------|
    | **Max Shelf Capacity** | Maximum units that can be stored on shelf |
    | **Unit Cost** | Cost per unit used for budget calculations |
    | **Max Budget per Order** | Maximum dollar amount per order |
    | **Perishability Days** | Product shelf life in days |
    | **Safety Stock Days** | Buffer stock to maintain |
    """)

# Page 4: Drift & Diagnostics
elif page == "🔍 Drift & Diagnostics":
    st.header("🔍 Drift Detection & Diagnostics")
    
    drift_events = load_drift_events()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Recent Drift Alerts")
        
        if len(drift_events) > 0:
            for _, event in drift_events.tail(10).iterrows():
                severity = float(event.get('details', '0').split('Severity: ')[-1].split(',')[0]) if 'Severity' in str(event.get('details', '')) else 0.5
                color = "🔴" if severity > 0.7 else "🟡" if severity > 0.4 else "🟢"
                st.markdown(f"{color} **{event['sku']}** - {event['timestamp'][:10]}")
        else:
            st.info("No drift events detected yet. Run drift analysis below.")
        
        st.markdown("---")
        st.subheader("Run Drift Check")
        check_sku = st.selectbox("Select SKU to analyze", skus)
        
        if st.button("🔍 Analyze Drift"):
            sku_df = df[df['sku'] == check_sku]
            
            if check_sku in engine.models:
                preds = engine.models[check_sku].predict(sku_df)
                actuals = sku_df['units_sold'].values
                residuals = (actuals - preds).tolist()
            else:
                residuals = np.random.normal(0, 5, len(sku_df)).tolist()
            
            drift_result = check_drift(check_sku, residuals)
            
            if drift_result['drift_detected']:
                st.warning(f"⚠️ Drift detected! Severity: {drift_result['severity']:.0%}")
                st.info("💡 Recommendation: Retrain the model with recent data to improve accuracy.")
            else:
                st.success("✅ No significant drift detected")
            
            st.session_state['current_drift'] = drift_result
    
    with col2:
        st.subheader("Diagnostic Report")
        
        if 'current_drift' in st.session_state:
            drift_info = st.session_state['current_drift']
            report = generate_drift_report(
                drift_info.get('sku', 'Unknown'),
                drift_info
            )
            st.markdown(report)
            
            # Error statistics chart
            st.subheader("Error Statistics")
            stats = drift_info.get('residual_stats', {})
            
            if stats:
                metrics_df = pd.DataFrame([
                    {'Metric': 'Mean Error', 'Value': stats.get('mean', 0)},
                    {'Metric': 'Std Dev', 'Value': stats.get('std', 0)},
                    {'Metric': 'MAE', 'Value': stats.get('mae', 0)}
                ])
                
                fig = px.bar(metrics_df, x='Metric', y='Value', 
                            title="Error Statistics",
                            color='Metric',
                            color_discrete_sequence=['#3498db', '#e74c3c', '#2ecc71'])
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select a SKU and run drift analysis to see diagnostics")

# Footer
st.markdown("---")
st.markdown("*Adaptive Demand Forecasting System v2.0 - With Continuous Learning*")
