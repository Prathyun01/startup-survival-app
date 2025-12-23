"""
Enhanced machine learning model for India-focused startup survival prediction.
Includes non-financial signals and explainable AI capabilities.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings('ignore')

class IndiaStartupSurvivalModel:
    """Enhanced model for India-focused startup survival prediction with explainable AI"""
    
    def __init__(self):
        # Core encoders
        self.sector_encoder = LabelEncoder()
        self.location_encoder = LabelEncoder()
        self.founder_background_encoder = LabelEncoder()
        self.incubator_encoder = LabelEncoder()
        self.grant_type_encoder = LabelEncoder()
        self.product_stage_encoder = LabelEncoder()
        self.customer_type_encoder = LabelEncoder()
        
        # Model - Updated for scikit-learn 1.4.0 compatibility
        self.model = RandomForestClassifier(
            n_estimators=100,  # Reduced from 200 for better compatibility
            max_depth=10,       # Reduced from 15
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,          # Use all CPU cores
            bootstrap=True,      # Enable bootstrapping
            oob_score=True      # Enable out-of-bag scoring
        )
        
        # Explainability
        self.feature_names = None
        self.is_fitted = False
        
    def fit(self, X, y):
        """Fit the model with preprocessing"""
        # Create a copy to avoid modifying original data
        X_processed = X.copy()
        
        # Encode categorical variables
        categorical_features = [
            'sector', 'location', 'founder_background', 'incubator_support', 
            'grant_type', 'product_stage', 'customer_type'
        ]
        for feature in categorical_features:
            if feature in X_processed.columns:
                encoder_name = f"{feature}_encoder"
                if hasattr(self, encoder_name):
                    encoder = getattr(self, encoder_name)
                    X_processed[f"{feature}_encoded"] = encoder.fit_transform(X_processed[feature])
        
        # Select features for training
        feature_columns = [
            'team_size', 'funding_inr_cr', 'team_diversity_score',
            'founder_experience_years', 'government_grants_inr_cr',
            'customer_count', 'revenue_inr_cr', 'burn_rate_months',
            'product_market_fit_score', 'customer_satisfaction_score',
            'incubator_support_encoded', 'grant_type_encoded',
            'sector_encoded', 'location_encoded', 'founder_background_encoded',
            'product_stage_encoded', 'customer_type_encoded'
        ]
        
        # Filter to available features
        available_features = [f for f in feature_columns if f in X_processed.columns]
        X_train = X_processed[available_features]
        self.feature_names = available_features
        
        # Fit the model
        self.model.fit(X_train, y)
        self.is_fitted = True
        
        return self
    
    def predict_proba(self, X):
        """Predict survival probability"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_processed = self._preprocess_features(X)
        return self.model.predict_proba(X_processed)
    
    def predict(self, X):
        """Predict survival class"""
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)
    
    def explain_prediction(self, X):
        """Generate explanations for predictions using feature importance"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before generating explanations")
        
        X_processed = self._preprocess_features(X)
        
        # Get feature importance for this prediction
        feature_importance = {}
        for i, feature in enumerate(self.feature_names):
            # Use feature importance weighted by feature value
            importance = self.model.feature_importances_[i]
            feature_value = X_processed.iloc[0, i] if len(X_processed) > 0 else 0
            
            # Normalize feature value for better interpretation
            if 'team_size' in feature:
                normalized_value = feature_value / 200  # Normalize by max team size
            elif 'funding' in feature:
                normalized_value = feature_value / 500  # Normalize by max funding
            elif 'experience' in feature:
                normalized_value = feature_value / 20   # Normalize by max experience
            elif 'diversity' in feature:
                normalized_value = feature_value        # Already 0-1
            elif 'grants' in feature:
                normalized_value = feature_value / 50   # Normalize by max grants
            elif 'customer_count' in feature:
                normalized_value = feature_value / 10000  # Normalize by max customers
            elif 'revenue' in feature:
                normalized_value = feature_value / 100   # Normalize by max revenue
            elif 'burn_rate' in feature:
                normalized_value = feature_value / 24    # Normalize by max burn rate
            elif 'product_market_fit' in feature:
                normalized_value = feature_value         # Already 0-1
            elif 'customer_satisfaction' in feature:
                normalized_value = feature_value         # Already 0-1
            else:
                normalized_value = feature_value / 10   # Default normalization
            
            feature_importance[feature] = importance * normalized_value
        
        # Sort by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'feature_importance': dict(sorted_features),
            'top_features': sorted_features[:5],  # Top 5 most important features
            'prediction_factors': self._get_prediction_factors(X_processed.iloc[0] if len(X_processed) > 0 else None)
        }
    
    def _get_prediction_factors(self, features):
        """Get human-readable prediction factors"""
        if features is None:
            return []
        
        factors = []
        feature_mapping = {
            'team_size': 'Team Size',
            'funding_inr_cr': 'Funding Amount',
            'team_diversity_score': 'Team Diversity',
            'founder_experience_years': 'Founder Experience',
            'government_grants_inr_cr': 'Government Grants',
            'customer_count': 'Customer Base',
            'revenue_inr_cr': 'Revenue',
            'burn_rate_months': 'Burn Rate',
            'product_market_fit_score': 'Product-Market Fit',
            'customer_satisfaction_score': 'Customer Satisfaction',
            'incubator_support_encoded': 'Incubator Support',
            'grant_type_encoded': 'Grant Type',
            'sector_encoded': 'Industry Sector',
            'location_encoded': 'Location',
            'founder_background_encoded': 'Founder Background',
            'product_stage_encoded': 'Product Stage',
            'customer_type_encoded': 'Customer Type'
        }
        
        for feature, value in features.items():
            if feature in feature_mapping:
                readable_name = feature_mapping[feature]
                factors.append(f"{readable_name}: {value:.2f}")
        
        return factors
    
    def get_feature_importance(self):
        """Get overall feature importance from the model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        importance_dict = {}
        for i, feature in enumerate(self.feature_names):
            importance_dict[feature] = self.model.feature_importances_[i]
        
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    def _preprocess_features(self, X):
        """Preprocess features for prediction"""
        X_processed = X.copy()
        
        # Encode categorical variables
        categorical_features = [
            'sector', 'location', 'founder_background', 'incubator_support', 
            'grant_type', 'product_stage', 'customer_type'
        ]
        for feature in categorical_features:
            if feature in X_processed.columns:
                encoder_name = f"{feature}_encoder"
                if hasattr(self, encoder_name):
                    encoder = getattr(self, encoder_name)
                    X_processed[f"{feature}_encoded"] = encoder.transform(X_processed[feature])
        
        # Select features for prediction
        available_features = [f for f in self.feature_names if f in X_processed.columns]
        return X_processed[available_features]

def create_india_startup_data(n_samples=2000):
    """Create comprehensive India-focused startup data with enhanced features"""
    
    np.random.seed(42)
    
    # India-specific sectors and locations
    india_sectors = [
        'Fintech', 'Edtech', 'Healthtech', 'E-commerce', 'SaaS', 'Agritech', 
        'Logistics', 'Gaming', 'AI/ML', 'CleanTech', 'D2C', 'B2B SaaS',
        'Insurtech', 'PropTech', 'FoodTech', 'TravelTech', 'HRTech'
    ]
    
    india_locations = [
        'Bengaluru', 'Mumbai', 'Delhi NCR', 'Hyderabad', 'Chennai', 'Pune',
        'Kolkata', 'Ahmedabad', 'Jaipur', 'Kochi', 'Indore', 'Chandigarh',
        'Lucknow', 'Patna', 'Bhopal', 'Vadodara', 'Surat', 'Nagpur'
    ]
    
    founder_backgrounds = [
        'IIT/IIM Graduate', 'Foreign University', 'Local University', 
        'Dropout', 'Corporate Executive', 'Academic', 'Serial Entrepreneur'
    ]
    
    incubators = [
        'Y Combinator', 'Techstars', '500 Startups', 'Sequoia Surge', 'Antler',
        'Axilor Ventures', 'TLabs', 'CIIE', 'NSRCEL', 'SINE IIT Bombay',
        'None', 'Local Incubator', 'University Incubator'
    ]
    
    grant_types = [
        'None', 'Startup India Seed Fund', 'ASPIRE', 'MUDRA', 'Stand-Up India',
        'PMEGP', 'NIDHI', 'BIRAC', 'State Government Grant', 'Corporate Grant'
    ]
    
    product_stages = [
        'Idea Stage', 'MVP Development', 'Beta Testing', 'Early Traction',
        'Product-Market Fit', 'Scaling', 'Mature Product'
    ]
    
    customer_types = [
        'B2B Enterprise', 'B2B SMB', 'B2C Mass Market', 'B2C Premium',
        'B2B2C', 'Marketplace', 'API/SaaS'
    ]
    
    # Generate synthetic data with enhanced features
    data = {
        'team_size': np.random.randint(1, 200, n_samples),
        'funding_inr_cr': np.random.uniform(0, 500, n_samples),
        'team_diversity_score': np.random.uniform(0.1, 1.0, n_samples),
        'founder_experience_years': np.random.uniform(0, 20, n_samples),
        'government_grants_inr_cr': np.random.uniform(0, 50, n_samples),
        'customer_count': np.random.randint(0, 10000, n_samples),
        'revenue_inr_cr': np.random.uniform(0, 100, n_samples),
        'burn_rate_months': np.random.uniform(0, 24, n_samples),
        'product_market_fit_score': np.random.uniform(0.1, 1.0, n_samples),
        'customer_satisfaction_score': np.random.uniform(0.1, 1.0, n_samples),
        'sector': np.random.choice(india_sectors, n_samples),
        'location': np.random.choice(india_locations, n_samples),
        'founder_background': np.random.choice(founder_backgrounds, n_samples),
        'incubator_support': np.random.choice(incubators, n_samples),
        'grant_type': np.random.choice(grant_types, n_samples),
        'product_stage': np.random.choice(product_stages, n_samples),
        'customer_type': np.random.choice(customer_types, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create synthetic survival labels based on comprehensive factors
    survival_prob = (
        # Financial factors (reduced weight from 40% to 25%)
        df['funding_inr_cr'] / 500 * 0.15 +
        df['government_grants_inr_cr'] / 50 * 0.10 +
        
        # Team factors (increased weight from 30% to 25%)
        df['team_size'] / 200 * 0.15 +
        df['team_diversity_score'] * 0.05 +
        df['founder_experience_years'] / 20 * 0.05 +
        
        # Product & Traction factors (NEW - 25% weight)
        df['customer_count'] / 10000 * 0.10 +
        df['revenue_inr_cr'] / 100 * 0.05 +
        df['product_market_fit_score'] * 0.05 +
        df['customer_satisfaction_score'] * 0.05 +
        
        # Location advantage (10%)
        (df['location'].isin(['Bengaluru', 'Mumbai', 'Delhi NCR', 'Hyderabad']) * 0.10) +
        
        # Sector advantage (5%)
        (df['sector'].isin(['Fintech', 'Edtech', 'SaaS', 'AI/ML', 'Healthtech']) * 0.05) +
        
        # Founder background advantage (5%)
        (df['founder_background'].isin(['IIT/IIM Graduate', 'Foreign University', 'Serial Entrepreneur']) * 0.05) +
        
        # Incubator advantage (5%)
        (df['incubator_support'].isin(['Y Combinator', 'Techstars', '500 Startups', 'Sequoia Surge']) * 0.05) +
        
        # Grant advantage (3%)
        (df['grant_type'] != 'None') * 0.03 +
        
        # Product stage advantage (2%)
        (df['product_stage'].isin(['Product-Market Fit', 'Scaling', 'Mature Product']) * 0.02)
    )
    
    # Add some randomness
    survival_prob += np.random.normal(0, 0.1, n_samples)
    survival_prob = np.clip(survival_prob, 0, 1)
    
    # Create binary survival labels
    df['survives'] = (survival_prob > 0.5).astype(int)
    
    return df

def create_benchmark_dataset():
    """Create a comprehensive benchmark dataset for research"""
    
    print("Creating India-focused startup benchmark dataset...")
    
    # Create main dataset
    df = create_india_startup_data(2000)
    
    # Add additional derived features
    df['funding_per_employee'] = df['funding_inr_cr'] / df['team_size']
    df['total_capital'] = df['funding_inr_cr'] + df['government_grants_inr_cr']
    df['revenue_per_customer'] = df['revenue_inr_cr'] / (df['customer_count'] + 1)  # Avoid division by zero
    df['runway_months'] = df['total_capital'] / (df['burn_rate_months'] + 1)  # Avoid division by zero
    df['is_tier1_city'] = df['location'].isin(['Bengaluru', 'Mumbai', 'Delhi NCR', 'Hyderabad']).astype(int)
    df['is_high_growth_sector'] = df['sector'].isin(['Fintech', 'Edtech', 'SaaS', 'AI/ML', 'Healthtech']).astype(int)
    df['has_prestigious_background'] = df['founder_background'].isin(['IIT/IIM Graduate', 'Foreign University']).astype(int)
    df['has_top_incubator'] = df['incubator_support'].isin(['Y Combinator', 'Techstars', '500 Startups', 'Sequoia Surge']).astype(int)
    df['has_government_support'] = (df['grant_type'] != 'None').astype(int)
    df['has_product_market_fit'] = (df['product_market_fit_score'] > 0.7).astype(int)
    df['has_customer_traction'] = (df['customer_count'] > 100).astype(int)
    df['is_revenue_generating'] = (df['revenue_inr_cr'] > 0).astype(int)
    
    # Save the dataset
    df.to_csv('india_startup_benchmark_dataset.csv', index=False)
    
    print(f"Dataset saved with {len(df)} samples and {len(df.columns)} features")
    print(f"Survival rate: {df['survives'].mean():.2%}")
    
    return df
