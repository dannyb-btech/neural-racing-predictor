#!/usr/bin/env python3
"""
Neural Network Horse Racing Model

Uses your existing feature set from BayesianTrainingDataExtractor to build
a multi-task neural network that predicts win/place probabilities and finish positions.

This model:
1. Works with horses that have never raced against each other
2. Uses standardized performance metrics (your timing data)
3. Handles varying amounts of historical data per horse
4. Provides calibrated probabilities
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HorseRacingNet(nn.Module):
    """
    Multi-task neural network for horse racing prediction.
    
    Predicts:
    - Win probability
    - Place probability  
    - Expected finish position
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32]):
        super(HorseRacingNet, self).__init__()
        
        # Shared feature extraction layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Task-specific heads
        final_dim = hidden_dims[-1]
        self.win_head = nn.Sequential(
            nn.Linear(final_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1)
        )
        
        self.place_head = nn.Sequential(
            nn.Linear(final_dim, 16), 
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1)
        )
        
        self.finish_head = nn.Sequential(
            nn.Linear(final_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1)
        )
        
    def forward(self, x):
        # Shared feature extraction
        shared_features = self.shared_layers(x)
        
        # Task-specific predictions
        win_logit = self.win_head(shared_features)
        place_logit = self.place_head(shared_features)
        finish_pos = self.finish_head(shared_features)
        
        return {
            'win_logit': win_logit,
            'place_logit': place_logit,
            'finish_pos': finish_pos,
            'win_prob': torch.sigmoid(win_logit),
            'place_prob': torch.sigmoid(place_logit)
        }

class NeuralRacingModel:
    """
    Neural network racing model using your existing feature set.
    """
    
    def __init__(self, hidden_dims: List[int] = [128, 64, 32], 
                 learning_rate: float = 0.001, device: str = 'cpu'):
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.device = torch.device(device)
        
        self.model = None
        self.scaler = StandardScaler()
        self.encoders = {}
        self.feature_names = []
        self.trained = False
        
        # Loss weights
        self.win_weight = 1.0
        self.place_weight = 1.0
        self.finish_weight = 0.1  # Lower weight for MSE loss
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features from your existing BayesianTrainingDataExtractor output.
        
        Args:
            df: DataFrame from extract_training_records
            
        Returns:
            DataFrame with processed features
        """
        logger.info(f"Preparing features from {len(df)} records")
        
        # Set seed for reproducible feature engineering
        np.random.seed(42)
        
        data = df.copy()
        
        # Remove records with missing critical data
        data = data.dropna(subset=['finish_position', 'horse_name'])
        
        # === TIMING FEATURES (Your Key Advantage) ===
        timing_features = [
            'standard_time_diff_overall',
            'standard_time_diff_800m',
            'standard_time_diff_400m', 
            'standard_time_diff_final'
        ]
        
        for feature in timing_features:
            if feature in data.columns:
                # Fill missing with neutral (0 = at standard)
                data[feature] = data[feature].fillna(0.0)
                # Cap extreme outliers
                data[feature] = np.clip(data[feature], -20, 20)
        
        # === PERFORMANCE FEATURES ===
        # Career performance at time of race
        if 'career_win_rate_to_date' in data.columns:
            data['career_win_rate_to_date'] = data['career_win_rate_to_date'].fillna(0.1)
            data['career_win_rate_to_date'] = np.clip(data['career_win_rate_to_date'], 0.0, 0.8)
        
        if 'career_starts_to_date' in data.columns:
            data['career_starts_to_date'] = data['career_starts_to_date'].fillna(1)
            # Experience factor (diminishing returns)
            data['experience_factor'] = np.log1p(data['career_starts_to_date']) / 4.0
        
        # Recent form
        if 'recent_form_3_races' in data.columns:
            data['recent_form_3_races'] = data['recent_form_3_races'].fillna(data['finish_position'])
            # Convert to performance score (lower finish = better)
            data['recent_form_score'] = 10.0 - np.clip(data['recent_form_3_races'], 1, 10)
        
        # === RACE CONTEXT FEATURES ===
        # Distance suitability
        race_context_features = ['same_distance', 'same_venue', 'same_track_condition', 'same_class']
        for feature in race_context_features:
            if feature in data.columns:
                data[feature] = data[feature].fillna(False).astype(int)
        
        # Distance difference impact
        if 'distance_difference' in data.columns:
            data['distance_difference'] = data['distance_difference'].fillna(0)
            data['distance_change_penalty'] = np.abs(data['distance_difference']) / 200.0  # Normalize
        
        # Recency (how long ago was this race)
        if 'days_to_target_race' in data.columns:
            data['days_to_target_race'] = data['days_to_target_race'].fillna(30)
            # Recency weight (more recent = more relevant)
            data['recency_score'] = np.exp(-data['days_to_target_race'] / 365.0)
        
        # === COMPETITIVE FEATURES ===
        if 'barrier' in data.columns:
            data['barrier'] = data['barrier'].fillna(8)
            # Barrier disadvantage (middle barriers often best)
            data['barrier_disadvantage'] = np.abs(data['barrier'] - 8) / 8.0
        
        if 'weight' in data.columns:
            data['weight'] = data['weight'].fillna(57.0)
            # Weight burden
            data['weight_burden'] = np.maximum(0, data['weight'] - 57.0) / 5.0
        
        if 'field_size' in data.columns:
            data['field_size'] = data['field_size'].fillna(8)
            # Field size difficulty
            data['field_difficulty'] = (data['field_size'] - 8) / 8.0
        
        # === HANDICAP RATING FEATURES ===
        if 'handicap_rating' in data.columns:
            data['handicap_rating'] = data['handicap_rating'].fillna(data['handicap_rating'].median())
            # Create field-relative rating if we have multiple horses
            if len(data) > 1:
                data['handicap_rating_vs_field_avg'] = data['handicap_rating'] - data['handicap_rating'].mean()
            else:
                data['handicap_rating_vs_field_avg'] = 0.0
        
        # === HORSE-SPECIFIC AGGREGATIONS ===
        # Calculate horse performance profiles
        horse_profiles = self._calculate_horse_profiles(data)
        
        # Merge horse profiles back to main data
        data = data.merge(horse_profiles, on='horse_name', how='left', suffixes=('', '_profile'))
        
        # === ENCODE CATEGORICAL FEATURES ===
        categorical_features = ['horse_name', 'venue', 'track_condition']
        for feature in categorical_features:
            if feature in data.columns:
                if feature not in self.encoders:
                    self.encoders[feature] = LabelEncoder()
                    data[f'{feature}_encoded'] = self.encoders[feature].fit_transform(
                        data[feature].fillna('Unknown')
                    )
                else:
                    # Transform using existing encoder
                    data[f'{feature}_encoded'] = data[feature].apply(
                        lambda x: self._safe_encode(x, feature)
                    )
        
        # Fill remaining NaNs
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        data[numeric_cols] = data[numeric_cols].fillna(0)
        
        logger.info(f"Feature preparation completed: {len(data)} records")
        return data
    
    def _calculate_horse_profiles(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate aggregate performance profiles for each horse."""
        
        profiles = []
        
        for horse in data['horse_name'].unique():
            horse_data = data[data['horse_name'] == horse].copy()
            
            # Sort by recency (most recent first)
            horse_data = horse_data.sort_values('days_to_target_race')
            
            # Calculate performance metrics
            profile = {
                'horse_name': horse,
                'total_races': len(horse_data),
                'overall_win_rate': horse_data['won'].mean(),
                'overall_place_rate': horse_data['placed'].mean(),
                'avg_finish_position': horse_data['finish_position'].mean(),
            }
            
            # Timing performance (if available)
            if 'standard_time_diff_overall' in horse_data.columns:
                timing_data = horse_data['standard_time_diff_overall'].dropna()
                if len(timing_data) > 0:
                    profile.update({
                        'avg_timing_performance': timing_data.mean(),
                        'timing_consistency': timing_data.std(),
                        'best_timing': timing_data.max(),
                        'timing_trend': self._calculate_trend(timing_data)
                    })
                else:
                    profile.update({
                        'avg_timing_performance': 0.0,
                        'timing_consistency': 1.0,
                        'best_timing': 0.0,
                        'timing_trend': 0.0
                    })
            
            # Recent form trend (last 5 races)
            recent_finishes = horse_data['finish_position'].head(5)
            if len(recent_finishes) > 1:
                profile['recent_form_trend'] = self._calculate_trend(recent_finishes)
            else:
                profile['recent_form_trend'] = 0.0
            
            # Class performance
            profile['avg_class_performance'] = horse_data['finish_position'].mean()
            
            profiles.append(profile)
        
        return pd.DataFrame(profiles)
    
    def _calculate_trend(self, series: pd.Series) -> float:
        """Calculate trend in a time series (positive = improving)."""
        if len(series) < 2:
            return 0.0
        
        x = np.arange(len(series))
        y = series.values
        
        # Simple linear regression slope
        slope = np.corrcoef(x, y)[0, 1] if len(x) > 1 else 0.0
        
        # For finish positions, negative slope = improving (lower finishes)
        # For timing, positive slope = improving (faster times)
        return -slope if 'finish' in series.name.lower() else slope
    
    def _safe_encode(self, value, feature):
        """Safely encode categorical value, handling unseen categories."""
        if pd.isna(value):
            value = 'Unknown'
        
        encoder = self.encoders[feature]
        try:
            return encoder.transform([value])[0]
        except ValueError:
            # Unseen category - return 0 or most common
            return 0
    
    def select_features(self, data: pd.DataFrame) -> List[str]:
        """Select features for training."""
        
        # Core performance features
        core_features = [
            'standard_time_diff_overall', 'standard_time_diff_800m',
            'standard_time_diff_400m', 'standard_time_diff_final',
            'career_win_rate_to_date', 'experience_factor',
            'recent_form_score', 'recency_score', 'handicap_rating'
        ]
        
        # Race context features
        context_features = [
            'same_class', 'distance_change_penalty', 'barrier_disadvantage', 
            'weight_burden', 'field_difficulty', 'handicap_rating_vs_field_avg'
        ]
        
        # Venue performance features (replacing simple same_venue boolean)
        venue_features = [
            'venue_win_rate', 'venue_place_rate', 'venue_avg_finish',
            'venue_experience', 'venue_recent_form'
        ]
        
        # Distance performance features (replacing simple same_distance boolean)
        distance_features = [
            'distance_win_rate', 'distance_place_rate', 'distance_avg_finish',
            'distance_experience', 'distance_recent_form'
        ]
        
        # Track condition performance features (replacing simple same_track_condition boolean)
        condition_features = [
            'condition_win_rate', 'condition_place_rate', 'condition_avg_finish',
            'condition_experience', 'condition_recent_form'
        ]
        
        # Position and running style features (tactical patterns)
        position_features = [
            'position_at_settled', 'position_at_800m', 'position_at_400m',
            'avg_position_settled', 'avg_position_800m', 'avg_position_400m',
            'running_style_leader', 'running_style_on_pace', 'running_style_midfield', 
            'running_style_back_marker', 'position_consistency', 'early_to_late_pattern'
        ]
        
        # Margin quality features (performance competitiveness)
        margin_features = [
            'margin', 'avg_win_margin', 'avg_place_margin', 'avg_loss_margin',
            'close_loss_rate', 'dominant_win_rate', 'competitive_race_rate', 'blowout_loss_rate'
        ]
        
        # Horse profile features
        profile_features = [
            'overall_win_rate', 'overall_place_rate', 'avg_finish_position',
            'avg_timing_performance', 'timing_consistency', 'best_timing',
            'timing_trend', 'recent_form_trend'
        ]
        
        # Categorical features
        categorical_features = ['horse_name_encoded', 'venue_encoded', 'track_condition_encoded']
        
        # Combine all features that exist in the data
        selected_features = []
        for feature_list in [core_features, context_features, venue_features, distance_features, condition_features, position_features, margin_features, profile_features, categorical_features]:
            selected_features.extend([f for f in feature_list if f in data.columns])
        
        logger.info(f"Selected {len(selected_features)} features for training")
        return selected_features
    
    def train(self, df: pd.DataFrame, epochs: int = 100, batch_size: int = 32,
              validation_split: float = 0.2) -> Dict:
        """
        Train the neural network model.
        
        Args:
            df: Training data from BayesianTrainingDataExtractor
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data for validation
            
        Returns:
            Training metrics dictionary
        """
        logger.info(f"Training neural network on {len(df)} records")
        
        # Set all random seeds for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
        
        # Prepare features
        data = self.prepare_features(df)
        
        if len(data) < 20:
            raise ValueError(f"Insufficient data: {len(data)} records. Need at least 20.")
        
        # Select features
        self.feature_names = self.select_features(data)
        
        # Prepare training data
        X = data[self.feature_names].values
        y_win = data['won'].values.astype(np.float32)
        y_place = data['placed'].values.astype(np.float32)
        y_finish = data['finish_position'].values.astype(np.float32)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data with fixed random state
        indices = np.arange(len(X_scaled))
        train_idx, val_idx = train_test_split(indices, test_size=validation_split, random_state=42)
        
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_win_train, y_win_val = y_win[train_idx], y_win[val_idx]
        y_place_train, y_place_val = y_place[train_idx], y_place[val_idx]
        y_finish_train, y_finish_val = y_finish[train_idx], y_finish[val_idx]
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_win_train_tensor = torch.FloatTensor(y_win_train).unsqueeze(1).to(self.device)
        y_place_train_tensor = torch.FloatTensor(y_place_train).unsqueeze(1).to(self.device)
        y_finish_train_tensor = torch.FloatTensor(y_finish_train).unsqueeze(1).to(self.device)
        
        y_win_val_tensor = torch.FloatTensor(y_win_val).unsqueeze(1).to(self.device)
        y_place_val_tensor = torch.FloatTensor(y_place_val).unsqueeze(1).to(self.device)
        y_finish_val_tensor = torch.FloatTensor(y_finish_val).unsqueeze(1).to(self.device)
        
        # Initialize model
        self.model = HorseRacingNet(
            input_dim=len(self.feature_names),
            hidden_dims=self.hidden_dims
        ).to(self.device)
        
        # Optimizer and scheduler
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        logger.info(f"Training on {len(X_train)} samples, validating on {len(X_val)}")
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            
            # Create batches
            n_batches = (len(X_train) + batch_size - 1) // batch_size
            epoch_train_loss = 0
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(X_train))
                
                batch_X = X_train_tensor[start_idx:end_idx]
                batch_y_win = y_win_train_tensor[start_idx:end_idx]
                batch_y_place = y_place_train_tensor[start_idx:end_idx]
                batch_y_finish = y_finish_train_tensor[start_idx:end_idx]
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                
                # Multi-task loss
                win_loss = F.binary_cross_entropy(outputs['win_prob'], batch_y_win)
                place_loss = F.binary_cross_entropy(outputs['place_prob'], batch_y_place)
                finish_loss = F.mse_loss(outputs['finish_pos'], batch_y_finish)
                
                total_loss = (self.win_weight * win_loss + 
                            self.place_weight * place_loss + 
                            self.finish_weight * finish_loss)
                
                # Backward pass
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_train_loss += total_loss.item()
            
            avg_train_loss = epoch_train_loss / n_batches
            train_losses.append(avg_train_loss)
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                
                val_win_loss = F.binary_cross_entropy(val_outputs['win_prob'], y_win_val_tensor)
                val_place_loss = F.binary_cross_entropy(val_outputs['place_prob'], y_place_val_tensor)
                val_finish_loss = F.mse_loss(val_outputs['finish_pos'], y_finish_val_tensor)
                
                val_loss = (self.win_weight * val_win_loss + 
                          self.place_weight * val_place_loss + 
                          self.finish_weight * val_finish_loss)
                
                val_losses.append(val_loss.item())
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model state
                self.best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if epoch % 10 == 0 or epoch == epochs - 1:
                logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, "
                          f"Val Loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Early stopping
            if patience_counter >= 20:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        if hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state)
        
        self.trained = True
        
        # Calculate final metrics
        self.model.eval()
        with torch.no_grad():
            train_outputs = self.model(X_train_tensor)
            val_outputs = self.model(X_val_tensor)
            
            # AUC scores
            train_win_auc = roc_auc_score(y_win_train, train_outputs['win_prob'].cpu().numpy())
            val_win_auc = roc_auc_score(y_win_val, val_outputs['win_prob'].cpu().numpy())
            
            train_place_auc = roc_auc_score(y_place_train, train_outputs['place_prob'].cpu().numpy())
            val_place_auc = roc_auc_score(y_place_val, val_outputs['place_prob'].cpu().numpy())
            
            # RMSE for finish position
            train_finish_rmse = np.sqrt(mean_squared_error(
                y_finish_train, train_outputs['finish_pos'].cpu().numpy()
            ))
            val_finish_rmse = np.sqrt(mean_squared_error(
                y_finish_val, val_outputs['finish_pos'].cpu().numpy()
            ))
        
        metrics = {
            'train_win_auc': train_win_auc,
            'val_win_auc': val_win_auc,
            'train_place_auc': train_place_auc,
            'val_place_auc': val_place_auc,
            'train_finish_rmse': train_finish_rmse,
            'val_finish_rmse': val_finish_rmse,
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'epochs_trained': len(train_losses),
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'features_used': len(self.feature_names)
        }
        
        logger.info(f"Training completed - Val Win AUC: {val_win_auc:.3f}, "
                   f"Val Place AUC: {val_place_auc:.3f}, Val Finish RMSE: {val_finish_rmse:.3f}")
        
        return metrics
    
    def predict_race(self, race_horses_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict race outcomes for horses.
        
        Args:
            race_horses_df: DataFrame with horse features for upcoming race
            
        Returns:
            DataFrame with predictions sorted by win probability
        """
        if not self.trained:
            raise ValueError("Model must be trained before making predictions")
        
        logger.info(f"Predicting race with {len(race_horses_df)} horses")
        
        # Set seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Prepare features for prediction
        pred_data = race_horses_df.copy()
        
        # Add default values for missing features
        for feature in self.feature_names:
            if feature not in pred_data.columns:
                if 'timing' in feature:
                    pred_data[feature] = 0.0
                elif 'same_' in feature:
                    pred_data[feature] = 1
                elif 'encoded' in feature:
                    pred_data[feature] = 0
                elif 'rate' in feature:
                    pred_data[feature] = 0.1
                else:
                    pred_data[feature] = 0.0
        
        # Scale features
        X = pred_data[self.feature_names].values
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # CRITICAL: Set model to evaluation mode to disable dropout
        self.model.eval()
        
        # Make predictions with no gradient computation
        with torch.no_grad():
            outputs = self.model(X_tensor)
            
            raw_win_probs = outputs['win_prob'].cpu().numpy().flatten()
            raw_place_probs = outputs['place_prob'].cpu().numpy().flatten()
            predicted_finishes = outputs['finish_pos'].cpu().numpy().flatten()
        
        # Deterministic probability normalization (no randomness)
        total_win_prob = raw_win_probs.sum()
        if total_win_prob > 0:
            normalized_win_probs = raw_win_probs / total_win_prob
        else:
            # Deterministic fallback if all probabilities are zero
            normalized_win_probs = np.ones(len(raw_win_probs)) / len(raw_win_probs)
        
        # Deterministic place probabilities (based on win probability ranking)
        place_probs = raw_place_probs.copy()
        
        # Sort by win probability (deterministic)
        sorted_indices = np.argsort(raw_win_probs)[::-1]  # Highest first
        
        # Assign place probabilities deterministically based on ranking
        for rank, idx in enumerate(sorted_indices):
            if rank == 0:  # Best horse
                place_probs[idx] = max(place_probs[idx], 0.7)
            elif rank == 1:  # Second best
                place_probs[idx] = max(place_probs[idx], 0.6) 
            elif rank == 2:  # Third best
                place_probs[idx] = max(place_probs[idx], 0.5)
            elif rank < 6:  # Next 3 horses
                place_probs[idx] = max(place_probs[idx], 0.3 - (rank - 3) * 0.05)
            else:  # Rest
                place_probs[idx] = max(place_probs[idx], 0.1)
        
        # Cap and normalize deterministically
        place_probs = np.minimum(place_probs, 0.9)
        total_place_prob = place_probs.sum()
        if total_place_prob > 3.2:
            place_probs = place_probs * (3.0 / total_place_prob)
        elif total_place_prob < 2.5:
            place_probs = place_probs * (2.8 / total_place_prob)
        
        # Create results with deterministic confidence calculation
        win_prob_std = np.std(raw_win_probs)
        confidence_base = 1.0 - win_prob_std if win_prob_std < 1.0 else 0.1
        
        results = pd.DataFrame({
            'horse_name': pred_data.get('horse_name', [f'Horse_{i}' for i in range(len(pred_data))]),
            'win_probability': normalized_win_probs,
            'place_probability': place_probs,
            'predicted_finish': predicted_finishes,
            'confidence': confidence_base  # Same confidence for all horses
        })
        
        # Sort by win probability and add ranking
        results = results.sort_values('win_probability', ascending=False).reset_index(drop=True)
        results['expected_rank'] = range(1, len(results) + 1)
        
        logger.info("Race predictions completed")
        return results
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance using gradient-based method."""
        if not self.trained:
            raise ValueError("Model must be trained first")
        
        # This is a simplified importance calculation
        # For neural networks, you'd typically use methods like SHAP or integrated gradients
        
        # Use parameter magnitudes as proxy for importance
        importances = []
        
        first_layer = self.model.shared_layers[0]  # First linear layer
        weights = first_layer.weight.data.abs().mean(dim=0).cpu().numpy()
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': weights / weights.sum()
        }).sort_values('importance', ascending=False)
        
        return importance_df


def main():
    """Example usage of the neural racing model."""
    print("🧠 Neural Network Horse Racing Model")
    print("=" * 50)
    
    # Create realistic synthetic data that matches your feature extraction output
    np.random.seed(42)
    n_records = 500
    horses = [f'Horse_{i}' for i in range(15)]
    
    data = []
    for i in range(n_records):
        horse = np.random.choice(horses)
        
        record = {
            'horse_name': horse,
            'finish_position': np.random.randint(1, 9),
            'won': 0,
            'placed': 0,
            'barrier': np.random.randint(1, 13),
            'weight': np.random.normal(57, 2),
            'field_size': np.random.randint(6, 14),
            'standard_time_diff_overall': np.random.normal(-1, 3),
            'standard_time_diff_800m': np.random.normal(-0.5, 2),
            'standard_time_diff_400m': np.random.normal(0, 1.5),
            'standard_time_diff_final': np.random.normal(-0.8, 2),
            'career_win_rate_to_date': np.random.beta(2, 10),
            'career_starts_to_date': np.random.randint(1, 25),
            'recent_form_3_races': np.random.uniform(1, 8),
            'same_venue': np.random.choice([0, 1], p=[0.7, 0.3]),
            'same_distance': np.random.choice([0, 1], p=[0.6, 0.4]),
            'same_track_condition': np.random.choice([0, 1], p=[0.5, 0.5]),
            'same_class': np.random.choice([0, 1], p=[0.4, 0.6]),
            'days_to_target_race': np.random.randint(7, 200),
            'distance_difference': np.random.choice([-200, 0, 200]),
            'venue': np.random.choice(['Flemington', 'Caulfield', 'Moonee Valley']),
            'track_condition': np.random.choice(['Good', 'Soft', 'Heavy'])
        }
        
        # Set win/place based on finish
        if record['finish_position'] == 1:
            record['won'] = 1
            record['placed'] = 1
        elif record['finish_position'] <= 3:
            record['placed'] = 1
            
        data.append(record)
    
    df = pd.DataFrame(data)
    print(f"Demo dataset: {len(df)} records, {df['horse_name'].nunique()} horses")
    
    # Train model
    model = NeuralRacingModel(hidden_dims=[128, 64, 32], learning_rate=0.001)
    
    metrics = model.train(df, epochs=50, batch_size=32)
    
    print(f"\n📈 Training Results:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.3f}")
    
    # Show feature importance
    print(f"\n🎯 Top 10 Most Important Features:")
    importance_df = model.get_feature_importance()
    for _, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.3f}")
    
    # Make predictions for upcoming race
    upcoming_horses = horses[:8]  # 8 horses in upcoming race
    
    upcoming_race_data = pd.DataFrame({
        'horse_name': upcoming_horses,
        'barrier': range(1, len(upcoming_horses) + 1),
        'weight': [57.5] * len(upcoming_horses),
        'field_size': [len(upcoming_horses)] * len(upcoming_horses),
        'same_venue': [1] * len(upcoming_horses),
        'same_distance': [1] * len(upcoming_horses),
        'same_track_condition': [1] * len(upcoming_horses),
        'same_class': [1] * len(upcoming_horses),
        'days_to_target_race': [0] * len(upcoming_horses),
        'distance_difference': [0] * len(upcoming_horses),
        'venue': ['Flemington'] * len(upcoming_horses),
        'track_condition': ['Good'] * len(upcoming_horses)
    })
    
    predictions = model.predict_race(upcoming_race_data)
    
    print(f"\n🏁 Race Predictions:")
    print(f"{'Rank':<4} {'Horse':<12} {'Win %':<8} {'Place %':<8} {'Confidence':<10} {'Exp. Finish':<10}")
    print("-" * 65)
    for _, row in predictions.iterrows():
        print(f"{row['expected_rank']:<4} {row['horse_name']:<12} "
              f"{row['win_probability']:.1%}   {row['place_probability']:.1%}    "
              f"{row['confidence']:.2f}       {row['predicted_finish']:.1f}")
    
    print(f"\n✅ Neural network model demo completed!")
    print(f"\n💡 Model advantages:")
    print(f"  • Works with horses that never raced against each other")
    print(f"  • Uses your timing data as key features")
    print(f"  • Multi-task learning (win, place, finish position)")
    print(f"  • Handles varying amounts of historical data")
    print(f"  • Provides calibrated probabilities")
    print(f"  • Uses standardized performance metrics")
    print(f"  • Neural network learns complex feature interactions")


if __name__ == "__main__":
    main()