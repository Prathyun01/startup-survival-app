from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import uvicorn
from typing import Optional, List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="India Startup Survival Prediction API",
    description="Enhanced API for predicting startup survival probability in India with explainable AI and comprehensive features",
    version="2.1.0"
)

# Load the pre-trained model
try:
    model = joblib.load('startup_survival_model.pkl')
    logger.info("Enhanced India-focused model loaded successfully")
except FileNotFoundError:
    logger.error("Model file not found. Please ensure startup_survival_model.pkl exists.")
    model = None

# Enhanced Pydantic model for request validation
class StartupData(BaseModel):
    team_size: int = Field(ge=1, le=1000, description="Number of team members")
    funding_inr_cr: float = Field(ge=0, le=10000, description="Funding raised in INR crores")
    team_diversity_score: float = Field(ge=0, le=1, description="Team diversity score (0-1)")
    founder_experience_years: float = Field(ge=0, le=50, description="Founder experience in years")
    government_grants_inr_cr: float = Field(ge=0, le=100, description="Government grants in INR crores")
    customer_count: int = Field(ge=0, le=100000, description="Number of customers/users")
    revenue_inr_cr: float = Field(ge=0, le=1000, description="Annual revenue in INR crores")
    burn_rate_months: float = Field(ge=0, le=60, description="Monthly burn rate in months of runway")
    product_market_fit_score: float = Field(ge=0, le=1, description="Product-market fit score (0-1)")
    customer_satisfaction_score: float = Field(ge=0, le=1, description="Customer satisfaction score (0-1)")
    sector: str = Field(description="Industry sector")
    location: str = Field(description="Company location")
    founder_background: str = Field(description="Founder educational background")
    incubator_support: str = Field(description="Incubator/accelerator support")
    grant_type: str = Field(description="Type of government grant")
    product_stage: str = Field(description="Current product development stage")
    customer_type: str = Field(description="Primary customer type")
    
    class Config:
        json_schema_extra = {
            "example": {
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
        }

# Enhanced Pydantic model for response
class PredictionResponse(BaseModel):
    survival_probability: float
    prediction: str
    confidence: str
    features_used: dict
    explanation: Optional[Dict[str, Any]] = None
    recommendations: List[str]
    risk_factors: List[str]
    strengths: List[str]

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "India Startup Survival Prediction API",
        "status": "healthy",
        "version": "2.1.0",
        "features": "Enhanced with explainable AI, comprehensive features, and India-specific factors"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_type": "IndiaStartupSurvivalModel" if model else None,
        "enhanced_features": True,
        "endpoints": [
            "/",
            "/health", 
            "/predict",
            "/explain",
            "/model-info",
            "/feature-importance",
            "/docs"
        ]
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_survival(startup_data: StartupData):
    """
    Predict startup survival probability based on comprehensive India-specific features
    
    **Enhanced Features:**
    - **Team**: Size, diversity, founder experience
    - **Financial**: Funding, grants, revenue, burn rate
    - **Product**: Market fit, customer satisfaction, product stage
    - **Traction**: Customer count, customer type
    - **Ecosystem**: Sector, location, incubator support, government grants
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Prepare features for prediction
        features = pd.DataFrame([startup_data.dict()])
        
        # Make prediction
        survival_prob = model.predict_proba(features)[0][1]  # Probability of survival
        
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
        
        # Generate explanation
        explanation = model.explain_prediction(features)
        
        # Generate comprehensive analysis
        analysis = analyze_startup_strengths_and_risks(startup_data, survival_prob, explanation)
        
        logger.info(f"Prediction made: {survival_prob:.3f} probability of survival")
        
        return PredictionResponse(
            survival_probability=round(survival_prob, 3),
            prediction=prediction,
            confidence=confidence,
            features_used=startup_data.dict(),
            explanation=explanation,
            recommendations=analysis['recommendations'],
            risk_factors=analysis['risk_factors'],
            strengths=analysis['strengths']
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/explain")
async def explain_prediction(startup_data: StartupData):
    """
    Get detailed explanation of prediction using enhanced feature analysis
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        features = pd.DataFrame([startup_data.dict()])
        explanation = model.explain_prediction(features)
        
        return {
            "explanation": explanation,
            "features_analyzed": startup_data.dict(),
            "feature_categories": {
                "team_factors": ["team_size", "team_diversity_score", "founder_experience_years"],
                "financial_factors": ["funding_inr_cr", "government_grants_inr_cr", "revenue_inr_cr", "burn_rate_months"],
                "product_factors": ["product_market_fit_score", "customer_satisfaction_score", "product_stage"],
                "traction_factors": ["customer_count", "customer_type"],
                "ecosystem_factors": ["sector", "location", "incubator_support", "grant_type"]
            }
        }
        
    except Exception as e:
        logger.error(f"Explanation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")

@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "model_type": type(model).__name__,
        "features": [
            "team_size", "funding_inr_cr", "team_diversity_score",
            "founder_experience_years", "government_grants_inr_cr",
            "customer_count", "revenue_inr_cr", "burn_rate_months",
            "product_market_fit_score", "customer_satisfaction_score",
            "sector", "location", "founder_background", 
            "incubator_support", "grant_type", "product_stage", "customer_type"
        ],
        "target": "survival_probability",
        "enhanced_features": True,
        "explainable_ai": True,
        "india_focused": True,
        "feature_categories": {
            "team": 3,
            "financial": 4,
            "product": 3,
            "traction": 2,
            "ecosystem": 4
        }
    }

@app.get("/feature-importance")
async def get_feature_importance():
    """Get overall feature importance from the model"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        importance = model.get_feature_importance()
        
        # Categorize features
        feature_categories = {
            "team": ["team_size", "team_diversity_score", "founder_experience_years"],
            "financial": ["funding_inr_cr", "government_grants_inr_cr", "revenue_inr_cr", "burn_rate_months"],
            "product": ["product_market_fit_score", "customer_satisfaction_score", "product_stage"],
            "traction": ["customer_count", "customer_type"],
            "ecosystem": ["sector", "location", "incubator_support", "grant_type"]
        }
        
        categorized_importance = {}
        for category, features in feature_categories.items():
            categorized_importance[category] = {
                feature: importance.get(feature, 0) 
                for feature in features 
                if feature in importance
            }
        
        return {
            "feature_importance": importance,
            "top_features": list(importance.items())[:5],
            "categorized_importance": categorized_importance
        }
        
    except Exception as e:
        logger.error(f"Feature importance error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Feature importance failed: {str(e)}")

def analyze_startup_strengths_and_risks(startup_data: StartupData, survival_prob: float, explanation: Dict) -> Dict:
    """Comprehensive analysis of startup strengths and risk factors"""
    
    strengths = []
    risk_factors = []
    recommendations = []
    
    # Analyze team factors
    if startup_data.team_size >= 10:
        strengths.append("👥 Strong team size for early-stage startup")
    elif startup_data.team_size < 5:
        risk_factors.append("👥 Small team size may limit execution capacity")
        recommendations.append("👥 Consider expanding team with key roles")
    
    if startup_data.team_diversity_score >= 0.7:
        strengths.append("🌈 High team diversity - good for innovation")
    elif startup_data.team_diversity_score < 0.3:
        risk_factors.append("🌈 Low team diversity may limit perspectives")
        recommendations.append("🌈 Focus on building diverse team composition")
    
    if startup_data.founder_experience_years >= 5:
        strengths.append("🎓 Experienced founder with industry knowledge")
    elif startup_data.founder_experience_years < 2:
        risk_factors.append("🎓 Limited founder experience may impact decision-making")
        recommendations.append("🎓 Seek mentorship and advisory support")
    
    # Analyze financial factors
    if startup_data.funding_inr_cr >= 20:
        strengths.append("💰 Adequate funding for current stage")
    elif startup_data.funding_inr_cr < 5:
        risk_factors.append("💰 Limited funding may constrain growth")
        recommendations.append("💰 Focus on fundraising or bootstrapping strategy")
    
    if startup_data.revenue_inr_cr > 0:
        strengths.append("💵 Revenue generating - good business model")
    else:
        risk_factors.append("💵 No revenue - need to validate business model")
        recommendations.append("💵 Focus on revenue generation and customer acquisition")
    
    if startup_data.burn_rate_months <= 12:
        strengths.append("⏰ Reasonable burn rate with good runway")
    elif startup_data.burn_rate_months > 18:
        risk_factors.append("⏰ High burn rate - limited runway")
        recommendations.append("⏰ Optimize costs and extend runway")
    
    # Analyze product factors
    if startup_data.product_market_fit_score >= 0.8:
        strengths.append("🎯 Strong product-market fit")
    elif startup_data.product_market_fit_score < 0.5:
        risk_factors.append("🎯 Weak product-market fit")
        recommendations.append("🎯 Focus on product-market fit validation")
    
    if startup_data.customer_satisfaction_score >= 0.8:
        strengths.append("😊 High customer satisfaction")
    elif startup_data.customer_satisfaction_score < 0.6:
        risk_factors.append("😊 Low customer satisfaction")
        recommendations.append("😊 Improve customer experience and support")
    
    # Analyze traction factors
    if startup_data.customer_count >= 100:
        strengths.append("👥 Good customer traction")
    elif startup_data.customer_count < 10:
        risk_factors.append("👥 Limited customer base")
        recommendations.append("👥 Focus on customer acquisition and growth")
    
    # Analyze ecosystem factors
    if startup_data.location in ['Bengaluru', 'Mumbai', 'Delhi NCR', 'Hyderabad']:
        strengths.append("🏙️ Located in tier-1 startup ecosystem")
    else:
        risk_factors.append("🏙️ Tier-2/3 location may limit access to resources")
        recommendations.append("🏙️ Consider benefits of tier-1 cities or build local ecosystem")
    
    if startup_data.incubator_support in ['Y Combinator', 'Techstars', '500 Startups', 'Sequoia Surge']:
        strengths.append("🚀 Top-tier incubator support")
    elif startup_data.incubator_support == 'None':
        risk_factors.append("🚀 No incubator support")
        recommendations.append("🚀 Apply to incubators for mentorship and networking")
    
    if startup_data.grant_type != 'None':
        strengths.append("🏛️ Government support through grants")
    else:
        risk_factors.append("🏛️ No government grant support")
        recommendations.append("🏛️ Explore Startup India and other government programs")
    
    # Overall recommendations based on survival probability
    if survival_prob >= 0.7:
        recommendations.insert(0, "🎉 Excellent! Your startup shows strong survival indicators.")
        recommendations.append("Focus on scaling and market expansion.")
        recommendations.append("Consider international expansion opportunities.")
    elif survival_prob >= 0.4:
        recommendations.insert(0, "⚠️ Moderate survival probability. Consider these improvements:")
    else:
        recommendations.insert(0, "🚨 Low survival probability. Critical areas to address:")
        recommendations.append("🔄 Reassess business model and market fit")
        recommendations.append("🔄 Consider pivot strategy if necessary")
    
    return {
        "strengths": strengths,
        "risk_factors": risk_factors,
        "recommendations": recommendations
    }

if __name__ == "__main__":
    uvicorn.run(
        "app_fastapi:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disabled auto-reload to prevent issues
        log_level="info"
    )
