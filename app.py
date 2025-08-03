import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
from preprocess import feature_engineering, preprocess

try:
    model = joblib.load('models/final_model.pkl')
    preprocessing_pipeline = joblib.load('models/preprocessing_pipeline.pkl')
except FileNotFoundError:
    st.error("Model or preprocessing pipeline file not found. Please check the 'models' directory.")
    st.stop()

AVG_HOUSE_VALUE = 206855.0

def validate_inputs(data):
    errors = []
    if not (-124.35 <= data['longitude'] <= -114.31):
        errors.append("Longitude must be between -124.35 and -114.31 (California range).")
    if not (32.54 <= data['latitude'] <= 41.95):
        errors.append("Latitude must be between 32.54 and 41.95 (California range).")
    if data['housing_median_age'] < 0 or data['housing_median_age'] > 100:
        errors.append("Housing Median Age must be between 0 and 100.")
    if data['total_rooms'] < 0:
        errors.append("Total Rooms cannot be negative.")
    if data['total_bedrooms'] < 0:
        errors.append("Total Bedrooms cannot be negative.")
    if data['population'] < 0:
        errors.append("Population cannot be negative.")
    if data['households'] < 0:
        errors.append("Households cannot be negative.")
    if data['median_income'] < 0:
        errors.append("Median Income cannot be negative.")
    if data['households'] == 0:  # Prevent division by zero
        errors.append("Households cannot be zero.")
    return errors

st.set_page_config(page_title="California Housing Price Predictor", layout="wide")

st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #135d94;
    }
    .metric-box {
        background-color: #e6f3ff;
        padding: 1rem;
        border-radius: 5px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🏠 California Housing Price Predictor</div>', unsafe_allow_html=True)
st.markdown("Enter housing details in the sidebar to predict the median house value and explore insights.")

with st.sidebar:
    st.header("📋 Input Housing Details")
    with st.form("input_form"):
        st.subheader("Geographical Information")
        longitude = st.number_input("Longitude", value=-122.23, format="%.6f")
        latitude = st.number_input("Latitude", value=37.88, format="%.6f")
        
        st.subheader("Housing Characteristics")
        housing_median_age = st.number_input("Housing Median Age", value=41.0, min_value=0.0, step=1.0)
        total_rooms = st.number_input("Total Rooms", value=880.0, min_value=0.0, step=1.0)
        total_bedrooms = st.number_input("Total Bedrooms", value=129.0, min_value=0.0, step=1.0)
        population = st.number_input("Population", value=322.0, min_value=0.0, step=1.0)
        households = st.number_input("Households", value=126.0, min_value=1.0, step=1.0)
        median_income = st.number_input("Median Income", value=8.3252, format="%.4f")
        
        st.subheader("Location Details")
        ocean_proximity = st.selectbox("Ocean Proximity", ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"])

        submit = st.form_submit_button("Predict House Value")

if submit:
    input_data = pd.DataFrame([{
        'longitude': longitude,
        'latitude': latitude,
        'housing_median_age': housing_median_age,
        'total_rooms': total_rooms,
        'total_bedrooms': total_bedrooms,
        'population': population,
        'households': households,
        'median_income': median_income,
        'ocean_proximity': ocean_proximity
    }])

    errors = validate_inputs(input_data.iloc[0])
    if errors:
        for error in errors:
            st.error(error)
    else:
        with st.container():
            st.markdown("### Prediction Results")
            try:
                processed_input = preprocess(input_data, preprocessing_pipeline)
                if processed_input is None:
                    st.stop()
                predicted_value = model.predict(processed_input)[0]

                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        label="🏡 Predicted House Value",
                        value=f"${predicted_value:,.2f}"
                    )

                with col2:
                    delta = predicted_value - AVG_HOUSE_VALUE
                    delta_color = "normal" if delta >= 0 else "inverse"

                    st.metric(
                        label="📊 Average House Value",
                        value=f"${AVG_HOUSE_VALUE:,.2f}",
                    )

                    if delta > 0:
                        st.success(f"🏆 Above average by ${delta:,.2f}")
                    elif delta < 0:
                        st.warning(f"📉 Below average by ${abs(delta):,.2f}")
                    else:
                        st.info("⚖️ Equal to the average value.")

                st.markdown("### Input Summary")
                st.dataframe(input_data.style.format({
                    'longitude': '{:.6f}',
                    'latitude': '{:.6f}',
                    'median_income': '{:.4f}'
                }))

                st.markdown("### Location on Map")
                map_data = pd.DataFrame({
                    'lat': [latitude],
                    'lon': [longitude]
                })
                st.map(map_data, zoom=10)

                try:
                    feature_importance = model.feature_importances_
                    feature_names = processed_input.columns
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': feature_importance
                    }).sort_values('Importance', ascending=False)

                    st.markdown("### Top 5 Feature Importance")
                    st.markdown("Chart showing the most influential features in the prediction:")
                    
                    fig = px.bar(
                        importance_df,  # Top 5 features
                        x='Feature',
                        y='Importance',
                        color='Feature',
                        color_discrete_sequence=px.colors.qualitative.Plotly,
                        title="Feature Importance",
                        labels={'Importance': 'Importance Score', 'Feature': 'Features'}
                    )
                    fig.update_layout(
                        showlegend=False,
                        yaxis_title="Importance",
                        xaxis_title="Features",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig, use_container_width=True)

                except AttributeError:
                    st.warning("Feature importance not available for this model type.")

            except Exception as e:
                st.error(f"Error during prediction: {e}")

st.markdown("---")
st.markdown("Built with Streamlit | Data processed using pre-trained machine learning model")