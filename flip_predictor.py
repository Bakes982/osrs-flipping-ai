#!/usr/bin/env python3
"""
OSRS Flip Predictor - ML model to predict long-term flip opportunities
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from typing import Dict, List, Tuple
import json

from flip_data_analyzer import FlipDataAnalyzer

class FlipPredictor:
    """ML model to predict flip profitability"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_trained = False
        
    def prepare_training_data(self, analyzer: FlipDataAnalyzer) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data from flip history
        
        Returns: (X features, y target)
        """
        print("Preparing training data...")
        
        # Get all item features
        features_df = analyzer.get_all_item_features()
        
        # Remove items with insufficient data
        features_df = features_df[features_df['flip_count'] >= 1].copy()
        
        # Select features for model
        feature_cols = [
            'avg_price',
            'avg_quantity',
            'flip_count',
            'avg_time_hours',
            'median_time_hours',
            'time_std',
            'success_rate',
            'avg_roi',
            'median_gp_hr',
            'peak_gp_hr',
            'profit_consistency',
            'time_consistency',
        ]
        
        # Handle missing values
        for col in feature_cols:
            features_df[col] = features_df[col].fillna(0)
        
        X = features_df[feature_cols].copy()
        
        # Target: average profit per flip
        y = features_df['avg_profit'].copy()
        
        self.feature_names = feature_cols
        
        print(f"✅ Prepared {len(X)} training samples with {len(feature_cols)} features")
        
        return X, y, features_df
    
    def train(self, analyzer: FlipDataAnalyzer):
        """Train the model on historical data"""
        print("="*80)
        print("TRAINING ML MODEL")
        print("="*80)
        
        # Prepare data
        X, y, full_df = self.prepare_training_data(analyzer)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print()
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest model
        print("Training Random Forest model...")
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        y_pred = self.model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"✅ Training complete!")
        print(f"   Train R² score: {train_score:.3f}")
        print(f"   Test R² score: {test_score:.3f}")
        print(f"   Mean Absolute Error: {mae:,.0f} GP")
        print()
        
        # Feature importance
        print("Top 10 Most Important Features:")
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for i, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']:25s}: {row['importance']:.3f}")
        print()
        
        self.is_trained = True
        
        return {
            'train_score': train_score,
            'test_score': test_score,
            'mae': mae,
            'feature_importance': feature_importance
        }
    
    def predict_profit(self, item_features: Dict) -> float:
        """Predict expected profit for an item"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        # Prepare features in correct order
        X = np.array([[
            item_features.get('avg_price', 0),
            item_features.get('avg_quantity', 0),
            item_features.get('flip_count', 0),
            item_features.get('avg_time_hours', 0),
            item_features.get('median_time_hours', 0),
            item_features.get('time_std', 0),
            item_features.get('success_rate', 0),
            item_features.get('avg_roi', 0),
            item_features.get('median_gp_hr', 0),
            item_features.get('peak_gp_hr', 0),
            item_features.get('profit_consistency', 0),
            item_features.get('time_consistency', 0),
        ]])
        
        # Scale and predict
        X_scaled = self.scaler.transform(X)
        predicted_profit = self.model.predict(X_scaled)[0]
        
        return predicted_profit
    
    def calculate_opportunity_score(self, item_features: Dict, predicted_profit: float) -> Dict:
        """
        Calculate overall opportunity score (0-100)
        
        Factors:
        - Predicted profit (50% weight)
        - Risk/consistency (30% weight)
        - Time efficiency (20% weight)
        """
        # Profit score (normalize to 0-100)
        profit_score = min(100, (predicted_profit / 500000) * 100)  # 500k = 100 points
        
        # Risk score (lower is better)
        risk_score = (
            item_features.get('profit_consistency', 0.5) * 0.5 +
            item_features.get('time_consistency', 0.5) * 0.3 +
            item_features.get('success_rate', 0.5) * 0.2
        ) * 100
        
        # Time efficiency score
        ideal_time = 12  # hours
        actual_time = item_features.get('avg_time_hours', 24)
        time_diff = abs(actual_time - ideal_time)
        time_score = max(0, 100 - (time_diff * 3))  # Penalize deviation from ideal
        
        # Weighted opportunity score
        opportunity_score = (
            profit_score * 0.5 +
            risk_score * 0.3 +
            time_score * 0.2
        )
        
        # Determine risk level
        if risk_score >= 70:
            risk_level = "LOW"
        elif risk_score >= 50:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"
        
        return {
            'opportunity_score': min(100, max(0, opportunity_score)),
            'profit_score': profit_score,
            'risk_score': risk_score,
            'time_score': time_score,
            'risk_level': risk_level,
            'predicted_profit': predicted_profit,
            'confidence': item_features.get('success_rate', 0.5) * 100
        }
    
    def find_opportunities(
        self, 
        analyzer: FlipDataAnalyzer,
        min_score: int = 20,
        min_time_hours: float = 4,
        max_time_hours: float = 168,
        min_price: int = 10000000,
        max_risk: str = "HIGH"
    ) -> pd.DataFrame:
        """
        Find long-term flip opportunities
        
        Args:
            analyzer: FlipDataAnalyzer instance
            min_score: Minimum opportunity score (0-100)
            min_time_hours: Minimum flip time
            max_time_hours: Maximum flip time
            min_price: Minimum item price
            max_risk: Maximum acceptable risk level
        
        Returns: DataFrame of opportunities
        """
        print("="*80)
        print("FINDING FLIP OPPORTUNITIES")
        print("="*80)
        print(f"Filters:")
        print(f"  Min Score: {min_score}")
        print(f"  Time Range: {min_time_hours}-{max_time_hours} hours")
        print(f"  Min Price: {min_price:,} GP")
        print(f"  Max Risk: {max_risk}")
        print()
        
        # Get all item features
        features_df = analyzer.get_all_item_features()
        
        opportunities = []
        
        for idx, row in features_df.iterrows():
            item_name = row['item_name']
            
            # Filter 1: Price
            if row['avg_price'] < min_price:
                continue
            
            # Filter 2: Time range
            if not (min_time_hours <= row['avg_time_hours'] <= max_time_hours):
                continue
            
            # Filter 3: Minimum historical success
            if row['flip_count'] < 1:
                continue
            
            # Predict profit
            try:
                predicted_profit = self.predict_profit(row.to_dict())
                
                # Filter 4: Must predict positive profit
                if predicted_profit <= 0:
                    continue
                
                # Calculate opportunity score
                scores = self.calculate_opportunity_score(row.to_dict(), predicted_profit)
                
                # Filter 5: Minimum score
                if scores['opportunity_score'] < min_score:
                    continue
                
                # Filter 6: Risk level
                risk_levels = {'LOW': 3, 'MEDIUM': 2, 'HIGH': 1}
                if risk_levels[scores['risk_level']] < risk_levels[max_risk]:
                    continue
                
                # Add to opportunities
                opportunities.append({
                    'item_name': item_name,
                    'opportunity_score': scores['opportunity_score'],
                    'predicted_profit': int(predicted_profit),
                    'risk_level': scores['risk_level'],
                    'confidence': scores['confidence'],
                    'avg_price': int(row['avg_price']),
                    'avg_time_hours': round(row['avg_time_hours'], 1),
                    'historical_avg_profit': int(row['avg_profit']),
                    'historical_gp_hr': int(row['avg_gp_hr']),
                    'flip_count': int(row['flip_count']),
                    'success_rate': round(row['success_rate'] * 100, 1),
                })
                
            except Exception as e:
                print(f"Warning: Could not process {item_name}: {e}")
                continue
        
        # Convert to DataFrame and sort
        opp_df = pd.DataFrame(opportunities)
        
        if len(opp_df) > 0:
            opp_df = opp_df.sort_values('opportunity_score', ascending=False)
        
        print(f"✅ Found {len(opp_df)} opportunities")
        print()
        
        return opp_df
    
    def save_model(self, filepath: str = "flip_predictor_model.pkl"):
        """Save trained model to disk"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        
        joblib.dump(model_data, filepath)
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath: str = "flip_predictor_model.pkl"):
        """Load trained model from disk"""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.is_trained = True
        
        print(f"✅ Model loaded from {filepath}")


# Test the predictor
if __name__ == "__main__":
    print("="*80)
    print("FLIP PREDICTOR TEST")
    print("="*80)
    print()
    
    # Load data
    analyzer = FlipDataAnalyzer('/mnt/user-data/uploads/flips.csv')
    
    # Create and train model
    predictor = FlipPredictor()
    results = predictor.train(analyzer)
    
    print()
    
    # Find opportunities
    opportunities = predictor.find_opportunities(
        analyzer,
        min_score=30,
        min_time_hours=4,
        max_time_hours=48,
        min_price=10000000,
        max_risk="HIGH"
    )
    
    if len(opportunities) > 0:
        print("="*80)
        print("TOP 10 LONG-TERM FLIP OPPORTUNITIES")
        print("="*80)
        print()
        
        for i, row in opportunities.head(10).iterrows():
            print(f"{int(row['opportunity_score']):3.0f}/100 | {row['item_name']:40s}")
            print(f"        Predicted Profit: {row['predicted_profit']:>10,} GP ({row['risk_level']} risk)")
            print(f"        Avg Time: {row['avg_time_hours']:.1f}h | Price: {row['avg_price']:,} GP")
            print(f"        Historical: {row['flip_count']} flips, {row['historical_avg_profit']:,} GP avg")
            print()
    
    print("✅ Test complete!")
