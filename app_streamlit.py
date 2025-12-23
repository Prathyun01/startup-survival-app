import streamlit as st
import pandas as pd
import joblib
import requests
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any

# Page configuration
st.set_page_config(
    page_title="India Startup Survival Predictor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-high {
        color: #28a745;
        font-weight: bold;
    }
    .prediction-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .prediction-low {
        color: #dc3545;
        font-weight: bold;
    }
    .feature-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the pre-trained model"""
    try:
        model = joblib.load('startup_survival_model.pkl')
        return model, True
    except FileNotFoundError:
        st.error("Model file not found. Please ensure startup_survival_model.pkl exists.")
        return None, False

def predict_survival_local(features: Dict[str, Any]) -> Dict[str, Any]:
    """Make prediction using local model"""
    model, loaded = load_model()
    if not loaded:
        return None
    
    try:
        # Prepare features
        df = pd.DataFrame([features])
        
        # Make prediction
        survival_prob = model.predict_proba(df)[0][1]
        
        # Get explanation
        explanation = model.explain_prediction(df)
        
        # Determine prediction and confidence
        if survival_prob >= 0.7:
            prediction = "High Survival Probability"
            confidence = "High"
        elif survival_prob >= 0.4:
            prediction = "Medium Survival Probability"
            confidence = "Medium"
        else:
            prediction = "Low Survival Probability"
            confidence = "Low"
        
        return {
            "survival_probability": round(survival_prob, 3),
            "prediction": prediction,
            "confidence": confidence,
            "features_used": features,
            "explanation": explanation
        }
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def predict_survival_api(features: Dict[str, Any]) -> Dict[str, Any]:
    """Make prediction using FastAPI endpoint"""
    try:
        response = requests.post(
            "http://localhost:8000/predict",
            json=features,
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.warning("⚠️ FastAPI server not running. Using local model instead.")
        return predict_survival_local(features)
    except Exception as e:
        st.error(f"API request failed: {str(e)}")
        return None

def create_gauge_chart(probability: float) -> go.Figure:
    """Create a gauge chart for survival probability"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Survival Probability (%)"},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 40], 'color': "lightgray"},
                {'range': [40, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def main():
    """Main application function"""
    # Header
    st.markdown('<h1 class="main-header">🚀 India Startup Survival Predictor</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Prediction", "About", "API Documentation"]
    )
    
    if page == "Prediction":
        show_prediction_page()
    elif page == "About":
        show_about_page()
    elif page == "API Documentation":
        show_api_docs_page()

def show_prediction_page():
    """Main prediction interface with enhanced features"""
    
    # Create two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Startup Information")
        
        # Input form with enhanced features
        with st.form("prediction_form"):
            st.markdown("#### 🏢 Team & Founder")
            team_size = st.number_input(
                "Team Size",
                min_value=1,
                max_value=200,
                value=15,
                help="Number of team members"
            )
            
            team_diversity_score = st.slider(
                "Team Diversity Score",
                min_value=0.1,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="Team diversity score (0.1-1.0)"
            )
            
            founder_experience_years = st.number_input(
                "Founder Experience (Years)",
                min_value=0.0,
                max_value=20.0,
                value=5.0,
                step=0.5,
                help="Founder experience in years"
            )
            
            founder_background = st.selectbox(
                "Founder Background",
                ["IIT/IIM Graduate", "Foreign University", "Local University", 
                 "Dropout", "Corporate Executive", "Academic", "Serial Entrepreneur"],
                help="Founder educational/professional background"
            )
            
            st.markdown("#### 💰 Financial Information")
            funding_inr_cr = st.number_input(
                "Funding (INR Crores)",
                min_value=0.0,
                max_value=500.0,
                value=25.0,
                step=0.1,
                help="Total funding raised in INR crores"
            )
            
            government_grants_inr_cr = st.number_input(
                "Government Grants (INR Crores)",
                min_value=0.0,
                max_value=50.0,
                value=3.0,
                step=0.1,
                help="Government grants received in INR crores"
            )
            
            revenue_inr_cr = st.number_input(
                "Annual Revenue (INR Crores)",
                min_value=0.0,
                max_value=100.0,
                value=2.5,
                step=0.1,
                help="Annual revenue in INR crores"
            )
            
            burn_rate_months = st.number_input(
                "Burn Rate (Months)",
                min_value=0.0,
                max_value=24.0,
                value=12.0,
                step=0.5,
                help="Monthly burn rate in months of runway"
            )
            
            st.markdown("#### 🎯 Product & Traction")
            customer_count = st.number_input(
                "Customer Count",
                min_value=0,
                max_value=10000,
                value=500,
                help="Number of customers/users"
            )
            
            product_market_fit_score = st.slider(
                "Product-Market Fit Score",
                min_value=0.1,
                max_value=1.0,
                value=0.8,
                step=0.1,
                help="Product-market fit score (0.1-1.0)"
            )
            
            customer_satisfaction_score = st.slider(
                "Customer Satisfaction Score",
                min_value=0.1,
                max_value=1.0,
                value=0.9,
                step=0.1,
                help="Customer satisfaction score (0.1-1.0)"
            )
            
            product_stage = st.selectbox(
                "Product Stage",
                ["Idea Stage", "MVP Development", "Beta Testing", "Early Traction",
                 "Product-Market Fit", "Scaling", "Mature Product"],
                help="Current product development stage"
            )
            
            customer_type = st.selectbox(
                "Customer Type",
                ["B2B Enterprise", "B2B SMB", "B2C Mass Market", "B2C Premium",
                 "B2B2C", "Marketplace", "API/SaaS"],
                help="Primary customer type"
            )
            
            st.markdown("#### 🌍 Ecosystem & Location")
            sector = st.selectbox(
                "Industry Sector",
                ["Fintech", "Edtech", "Healthtech", "E-commerce", "SaaS", "Agritech", 
                 "Logistics", "Gaming", "AI/ML", "CleanTech", "D2C", "B2B SaaS",
                 "Insurtech", "PropTech", "FoodTech", "TravelTech", "HRTech"],
                help="Primary industry sector"
            )
            
            location = st.selectbox(
                "Location",
                ["Bengaluru", "Mumbai", "Delhi NCR", "Hyderabad", "Chennai", "Pune",
                 "Kolkata", "Ahmedabad", "Jaipur", "Kochi", "Indore", "Chandigarh",
                 "Lucknow", "Patna", "Bhopal", "Vadodara", "Surat", "Nagpur"],
                help="Company headquarters location"
            )
            
            incubator_support = st.selectbox(
                "Incubator/Accelerator Support",
                ["None", "Y Combinator", "Techstars", "500 Startups", "Sequoia Surge", "Antler",
                 "Axilor Ventures", "TLabs", "CIIE", "NSRCEL", "SINE IIT Bombay",
                 "Local Incubator", "University Incubator"],
                help="Incubator or accelerator support"
            )
            
            grant_type = st.selectbox(
                "Government Grant Type",
                ["None", "Startup India Seed Fund", "ASPIRE", "MUDRA", "Stand-Up India",
                 "PMEGP", "NIDHI", "BIRAC", "State Government Grant", "Corporate Grant"],
                help="Type of government grant received"
            )
            
            # Prediction method selection
            use_api = st.checkbox("Use FastAPI Server (if available)", value=False)
            
            submitted = st.form_submit_button("🚀 Predict Survival Probability")
    
    with col2:
        st.subheader("📈 Prediction Results")
        
        if submitted:
            # Prepare features
            features = {
                "team_size": team_size,
                "funding_inr_cr": funding_inr_cr,
                "team_diversity_score": team_diversity_score,
                "founder_experience_years": founder_experience_years,
                "government_grants_inr_cr": government_grants_inr_cr,
                "customer_count": customer_count,
                "revenue_inr_cr": revenue_inr_cr,
                "burn_rate_months": burn_rate_months,
                "product_market_fit_score": product_market_fit_score,
                "customer_satisfaction_score": customer_satisfaction_score,
                "sector": sector,
                "location": location,
                "founder_background": founder_background,
                "incubator_support": incubator_support,
                "grant_type": grant_type,
                "product_stage": product_stage,
                "customer_type": customer_type
            }
            
            # Show loading spinner
            with st.spinner("Analyzing startup data..."):
                # Make prediction
                if use_api:
                    result = predict_survival_api(features)
                else:
                    result = predict_survival_local(features)
            
            if result:
                # Display results
                st.markdown("### Prediction Results")
                
                # Gauge chart
                gauge_fig = create_gauge_chart(result["survival_probability"])
                st.plotly_chart(gauge_fig, use_container_width=True)
                
                # Metrics
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.metric(
                        "Survival Probability",
                        f"{result['survival_probability']:.1%}",
                        delta=f"{result['survival_probability'] - 0.5:.1%}"
                    )
                
                with col_b:
                    st.metric(
                        "Prediction",
                        result["prediction"],
                        delta=result["confidence"]
                    )
                
                with col_c:
                    st.metric(
                        "Confidence",
                        result["confidence"],
                        delta="Model Confidence"
                    )
                
                # Show explanation if available
                if "explanation" in result and result["explanation"]:
                    st.markdown("### 🔍 Prediction Explanation")
                    
                    # Top features
                    if "top_features" in result["explanation"]:
                        st.markdown("#### Top 5 Most Important Features:")
                        top_features = result["explanation"]["top_features"]
                        
                        for i, (feature, importance) in enumerate(top_features[:5], 1):
                            st.markdown(f"**{i}.** {feature}: {importance:.3f}")
                    
                    # Feature importance chart
                    if "feature_importance" in result["explanation"]:
                        st.markdown("#### Feature Importance Breakdown:")
                        importance_data = result["explanation"]["feature_importance"]
                        
                        if importance_data:
                            # Create feature importance chart
                            features_list = list(importance_data.keys())
                            importance_values = list(importance_data.values())
                            
                            fig = px.bar(
                                x=importance_values,
                                y=features_list,
                                orientation='h',
                                title="Feature Importance for This Prediction",
                                labels={'x': 'Importance Score', 'y': 'Features'}
                            )
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                
                # Show prediction factors
                if "explanation" in result and "prediction_factors" in result["explanation"]:
                    st.markdown("### 📊 Prediction Factors")
                    factors = result["explanation"]["prediction_factors"]
                    
                    for factor in factors[:10]:  # Show first 10 factors
                        st.info(f"• {factor}")
                
                # Save prediction history
                if 'prediction_history' not in st.session_state:
                    st.session_state.prediction_history = []
                
                st.session_state.prediction_history.append({
                    'timestamp': datetime.now(),
                    'features': features,
                    'result': result
                })

def show_about_page():
    """About page with project information"""
    st.subheader("📖 About India Startup Survival Predictor")
    
    st.markdown("""
    This application uses advanced machine learning to predict the survival probability of Indian startups based on comprehensive metrics.
    
    ### 🎯 Purpose
    The India Startup Survival Predictor helps entrepreneurs, investors, and analysts assess the likelihood of a startup's success
    based on quantifiable factors including team composition, financial metrics, product traction, and ecosystem factors.
    
    ### 🔬 Methodology
    - **Model**: Enhanced Random Forest with 200 estimators
    - **Features**: 16 core features across 5 categories
    - **Output**: Survival probability with detailed explanations
    - **Explainability**: Feature importance and prediction breakdowns
    
    ### 📊 Features Analyzed
    
    #### 🏢 Team Factors
    - **Team Size**: Number of team members (1-200)
    - **Team Diversity**: Diversity score (0.1-1.0)
    - **Founder Experience**: Years of experience (0-20)
    - **Founder Background**: Educational/professional background
    
    #### 💰 Financial Factors
    - **Funding**: Total funding raised in INR crores
    - **Government Grants**: Grant amount in INR crores
    - **Revenue**: Annual revenue in INR crores
    - **Burn Rate**: Monthly burn rate in months
    
    #### 🎯 Product Factors
    - **Product-Market Fit**: Score (0.1-1.0)
    - **Customer Satisfaction**: Score (0.1-1.0)
    - **Product Stage**: Development stage
    
    #### 📈 Traction Factors
    - **Customer Count**: Number of customers/users
    - **Customer Type**: B2B, B2C, Marketplace, etc.
    
    #### 🌍 Ecosystem Factors
    - **Sector**: 17 India-specific sectors
    - **Location**: 18 major Indian cities
    - **Incubator Support**: Incubator/accelerator participation
    - **Government Support**: Grant types and programs
    
    ### ⚠️ Disclaimer
    This tool provides predictions based on historical data patterns and should not be considered as financial advice.
    Startup success depends on many factors beyond those analyzed here.
    """)
    
    # Model information
    model, loaded = load_model()
    if loaded:
        st.success("✅ Enhanced model loaded successfully")
        st.info(f"Model type: {type(model).__name__}")
        
        # Show model capabilities
        try:
            feature_importance = model.get_feature_importance()
            if feature_importance:
                st.markdown("#### 🏆 Top 5 Most Important Features:")
                top_features = list(feature_importance.items())[:5]
                for i, (feature, importance) in enumerate(top_features, 1):
                    st.markdown(f"**{i}.** {feature}: {importance:.3f}")
        except:
            st.info("Feature importance information available during predictions")
    else:
        st.error("❌ Model not available")

def show_api_docs_page():
    """API documentation page"""
    st.subheader("🔌 API Documentation")
    
    st.markdown("""
    ### FastAPI Endpoints
    
    The application provides a RESTful API for programmatic access to predictions.
    
    #### Base URL
    ```
    http://localhost:8000
    ```
    
    #### Endpoints
    
    **1. Health Check**
    ```
    GET /
    ```
    
    **2. Predict Survival**
    ```
    POST /predict
    ```
    
    **Request Body (Enhanced Features):**
    ```json
    {
        "team_size": 15,
        "funding_inr_cr": 25.0,
        "team_diversity_score": 0.7,
        "founder_experience_years": 5.0,
        "government_grants_inr_cr": 3.0,
        "customer_count": 500,
        "revenue_inr_cr": 2.5,
        "burn_rate_months": 12.0,
        "product_market_fit_score": 0.8,
        "customer_satisfaction_score": 0.9,
        "sector": "SaaS",
        "location": "Bengaluru",
        "founder_background": "IIT/IIM Graduate",
        "incubator_support": "Y Combinator",
        "grant_type": "Startup India Seed Fund",
        "product_stage": "Product-Market Fit",
        "customer_type": "B2B SMB"
    }
    ```
    
    **Response:**
    ```json
    {
        "survival_probability": 0.75,
        "prediction": "High Survival Probability",
        "confidence": "High",
        "features_used": {...},
        "explanation": {
            "feature_importance": {...},
            "top_features": [...],
            "prediction_factors": [...]
        },
        "recommendations": [...],
        "risk_factors": [...],
        "strengths": [...]
    }
    ```
    
    **3. Model Information**
    ```
    GET /model-info
    ```
    
    **4. Feature Importance**
    ```
    GET /feature-importance
    ```
    
    **5. Interactive Documentation**
    ```
    GET /docs
    ```
    
    ### Example Usage
    
    ```python
    import requests
    
    # Make prediction
    response = requests.post(
        "http://localhost:8000/predict",
        json={
            "team_size": 15,
            "funding_inr_cr": 25.0,
            "team_diversity_score": 0.7,
            "founder_experience_years": 5.0,
            "government_grants_inr_cr": 3.0,
            "customer_count": 500,
            "revenue_inr_cr": 2.5,
            "burn_rate_months": 12.0,
            "product_market_fit_score": 0.8,
            "customer_satisfaction_score": 0.9,
            "sector": "SaaS",
            "location": "Bengaluru",
            "founder_background": "IIT/IIM Graduate",
            "incubator_support": "Y Combinator",
            "grant_type": "Startup India Seed Fund",
            "product_stage": "Product-Market Fit",
            "customer_type": "B2B SMB"
        }
    )
    
    result = response.json()
    print(f"Survival probability: {result['survival_probability']:.1%}")
    ```
    """)

# Main execution
if __name__ == "__main__":
    main()
