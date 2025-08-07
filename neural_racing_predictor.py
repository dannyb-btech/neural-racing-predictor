#!/usr/bin/env python3
"""
Complete Neural Racing Predictor

Combines your existing feature extraction with the neural network model.
Usage: python neural_racing_predictor.py 2025-07-16 "Flemington" 7 --api-key YOUR_KEY

This script:
1. Extracts historical training data using your BayesianTrainingDataExtractor
2. Trains a neural network model on that data
3. Generates predictions with calibrated probabilities
4. Handles horses that have never raced against each other
"""

import argparse
import pandas as pd
import numpy as np
import logging
import sys
import os
from datetime import datetime
import json
import torch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import your existing modules
from racing_com_api_client import RacingComAPIClient
from feature_creation import BayesianTrainingDataExtractor, RaceTarget
from neural_racing_model import NeuralRacingModel

# Optional Cosmos DB import
try:
    from cosmos_db_client import CosmosDBClient
    COSMOS_DB_AVAILABLE = True
except ImportError:
    COSMOS_DB_AVAILABLE = False
    logger.warning("Cosmos DB client not available - predictions will only be saved locally")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_race_summary(race_info: dict, horse_count: int):
    """Print summary of the upcoming race."""
    print("\n" + "="*60)
    print("🏁 UPCOMING RACE SUMMARY")
    print("="*60)
    
    print(f"Date: {race_info['date']}")
    print(f"Venue: {race_info['venue']}")
    print(f"Race: {race_info['race_number']}")
    print(f"Distance: {race_info.get('distance', 'Unknown')}m")
    print(f"Track: {race_info.get('track_condition', 'Unknown')}")
    print(f"Class: {race_info.get('class', 'Unknown')}")
    print(f"Field Size: {horse_count} horses")

def print_training_summary(df: pd.DataFrame, target_horses: list):
    """Print summary of training data."""
    print("\n" + "="*60)
    print("📊 TRAINING DATA SUMMARY") 
    print("="*60)
    
    print(f"Total historical records: {len(df):,}")
    print(f"Target race horses: {len(target_horses)}")
    print(f"Date range: {df['race_date'].min()} to {df['race_date'].max()}")
    
    print(f"\nPerformance Distribution:")
    print(f"  Win rate: {df['won'].mean():.1%}")
    print(f"  Place rate: {df['placed'].mean():.1%}")
    print(f"  Average finish: {df['finish_position'].mean():.1f}")
    
    # Timing data analysis
    timing_cols = [col for col in df.columns if 'standard_time_diff' in col]
    if timing_cols:
        timing_data = df['standard_time_diff_overall'].dropna()
        if len(timing_data) > 0:
            print(f"\nTiming Data (lengths vs standard):")
            print(f"  Records with timing: {len(timing_data):,} ({len(timing_data)/len(df):.1%})")
            print(f"  Average: {timing_data.mean():+.2f}L")
            print(f"  Range: {timing_data.min():+.1f}L to {timing_data.max():+.1f}L")
    
    # Horse-specific stats
    horse_stats = df.groupby('horse_name').agg({
        'finish_position': 'count',
        'won': 'sum', 
        'placed': 'sum',
        'standard_time_diff_overall': lambda x: x.dropna().mean() if len(x.dropna()) > 0 else None
    }).round(2)
    
    print(f"\nHorse Statistics (Top 10 by races):")
    print(f"{'Horse':<20} {'Races':<6} {'Wins':<5} {'Places':<7} {'Avg Timing':<10}")
    print("-" * 60)
    
    top_horses = horse_stats.sort_values('finish_position', ascending=False).head(10)
    for horse in top_horses.index:
        if horse in target_horses:
            stats = top_horses.loc[horse]
            timing = f"{stats['standard_time_diff_overall']:+.1f}L" if pd.notna(stats['standard_time_diff_overall']) else "N/A"
            print(f"{horse:<20} {stats['finish_position']:<6} {stats['won']:<5} {stats['placed']:<7} {timing:<10}")

def print_model_results(metrics: dict, predictions: pd.DataFrame, race_info: dict = None):
    """Print model performance and predictions."""
    print("\n" + "="*60)
    print("🧠 NEURAL NETWORK RESULTS")
    print("="*60)
    
    print(f"Model Performance:")
    print(f"  Win prediction AUC: {metrics['val_win_auc']:.3f}")
    print(f"  Place prediction AUC: {metrics['val_place_auc']:.3f}")
    print(f"  Finish position RMSE: {metrics['val_finish_rmse']:.3f}")
    print(f"  Training samples: {metrics['training_samples']:,}")
    print(f"  Validation samples: {metrics['validation_samples']:,}")
    print(f"  Features used: {metrics['features_used']}")
    print(f"  Epochs trained: {metrics['epochs_trained']}")
    
    # Model quality assessment
    win_auc = metrics['val_win_auc']
    place_auc = metrics['val_place_auc']
    finish_rmse = metrics['val_finish_rmse']
    
    print(f"\nModel Quality Assessment:")
    if win_auc > 0.7:
        print(f"  ✅ Win Prediction: Excellent (AUC > 0.7)")
    elif win_auc > 0.6:
        print(f"  ✅ Win Prediction: Good (AUC > 0.6)")
    elif win_auc > 0.55:
        print(f"  ⚠️  Win Prediction: Fair (AUC > 0.55)")
    else:
        print(f"  ❌ Win Prediction: Poor (AUC < 0.55)")
    
    if place_auc > 0.7:
        print(f"  ✅ Place Prediction: Excellent (AUC > 0.7)")
    elif place_auc > 0.6:
        print(f"  ✅ Place Prediction: Good (AUC > 0.6)")
    elif place_auc > 0.55:
        print(f"  ⚠️  Place Prediction: Fair (AUC > 0.55)")
    else:
        print(f"  ❌ Place Prediction: Poor (AUC < 0.55)")
    
    if finish_rmse < 2.0:
        print(f"  ✅ Finish Position: Excellent (RMSE < 2.0)")
    elif finish_rmse < 2.5:
        print(f"  ✅ Finish Position: Good (RMSE < 2.5)")
    elif finish_rmse < 3.0:
        print(f"  ⚠️  Finish Position: Fair (RMSE < 3.0)")
    else:
        print(f"  ❌ Finish Position: Poor (RMSE > 3.0)")
    
    # Show race information in header
    if race_info:
        print(f"\n🏆 RACE {race_info.get('race_number', 'N/A')} PREDICTIONS:")
        print(f"📍 {race_info.get('venue', 'Unknown Venue')} - {race_info.get('distance', 'Unknown')}m - {race_info.get('track_condition', 'Unknown')} Track")
    else:
        print(f"\n🏆 RACE PREDICTIONS:")
    
    print(f"\n{'Rank':<4} {'#':<3} {'Horse':<25} {'Win %':<8} {'Place %':<8} {'Confidence':<10} {'Exp. Finish':<10}")
    print("-" * 85)
    
    for _, row in predictions.iterrows():
        # Get saddlecloth number from the predictions data
        horse_number = row.get('saddlecloth_number', row.get('horse_number', row['expected_rank']))
        
        print(f"{row['expected_rank']:<4} {horse_number:<3} {row['horse_name']:<25} "
              f"{row['win_probability']:.1%}    {row['place_probability']:.1%}    "
              f"{row['confidence']:.2f}       {row['predicted_finish']:.1f}")
    
    # Show top 3 picks
    top_3 = predictions.head(3)
    print(f"\n🥇 TOP 3 SELECTIONS:")
    for i, (_, horse) in enumerate(top_3.iterrows(), 1):
        medal = ["🥇", "🥈", "🥉"][i-1]
        horse_number = horse.get('saddlecloth_number', horse.get('horse_number', i))
        print(f"  {medal} #{horse_number} {horse['horse_name']} - {horse['win_probability']:.1%} win chance, "
              f"{horse['place_probability']:.1%} place chance")
    
    # Probability validation
    total_win_prob = predictions['win_probability'].sum()
    total_place_prob = predictions['place_probability'].sum()
    print(f"\n📊 Probability Validation:")
    print(f"  Total win probabilities: {total_win_prob:.1%} (should be 100%)")
    print(f"  Total place probabilities: {total_place_prob:.1f} (should be ~3.0)")

def calculate_distance_penalty(profile: pd.Series, race_distance: int, training_df: pd.DataFrame, horse_name: str) -> float:
    """Calculate distance change penalty based on horse's historical performance at different distances."""
    try:
        # Get horse's historical distances and performance
        horse_races = training_df[training_df['horse_name'] == horse_name]
        if len(horse_races) == 0:
            return 0.0
        
        # Find the distance where horse performs best (lowest average finish)
        distance_performance = horse_races.groupby('distance')['finish_position'].mean()
        
        if len(distance_performance) == 0:
            return 0.0
        
        optimal_distance = distance_performance.idxmin()
        distance_diff = abs(race_distance - optimal_distance)
        
        # Penalty increases with distance from optimal (0-200m = 0, >400m = significant penalty)
        if distance_diff <= 100:
            return 0.0
        elif distance_diff <= 200:
            return 0.1
        elif distance_diff <= 400:
            return 0.2
        else:
            return 0.3
            
    except Exception:
        return 0.0

def calculate_venue_performance(horse_name: str, target_venue: str, training_df: pd.DataFrame) -> dict:
    """Calculate venue-specific performance metrics for a horse."""
    
    # Filter for this horse's races at the target venue
    horse_races = training_df[training_df['horse_name'] == horse_name]
    venue_races = horse_races[horse_races['venue'].str.lower().str.contains(target_venue.lower(), na=False)]
    
    if len(venue_races) == 0:
        # No experience at this venue
        return {
            'venue_starts': 0,
            'venue_win_rate': 0.0,
            'venue_place_rate': 0.0,
            'venue_avg_finish': 5.0,  # Default middle finish
            'venue_experience': 0.0,
            'venue_recent_form': 5.0
        }
    
    # Calculate venue-specific metrics
    venue_starts = len(venue_races)
    venue_wins = venue_races['won'].sum()
    venue_places = venue_races['placed'].sum()
    venue_win_rate = venue_wins / venue_starts
    venue_place_rate = venue_places / venue_starts
    venue_avg_finish = venue_races['finish_position'].mean()
    venue_experience = np.log1p(venue_starts)  # log(starts + 1) for diminishing returns
    
    # Recent form at venue (last 3 races or all if fewer)
    recent_venue_races = venue_races.tail(3)
    venue_recent_form = recent_venue_races['finish_position'].mean()
    
    return {
        'venue_starts': venue_starts,
        'venue_win_rate': venue_win_rate,
        'venue_place_rate': venue_place_rate,
        'venue_avg_finish': venue_avg_finish,
        'venue_experience': venue_experience,
        'venue_recent_form': venue_recent_form
    }

def calculate_distance_performance(horse_name: str, target_distance: int, training_df: pd.DataFrame) -> dict:
    """Calculate distance-specific performance metrics for a horse."""
    
    # Filter for this horse's races at the target distance
    horse_races = training_df[training_df['horse_name'] == horse_name]
    distance_races = horse_races[horse_races['distance'] == target_distance]
    
    if len(distance_races) == 0:
        # No experience at this distance
        return {
            'distance_starts': 0,
            'distance_win_rate': 0.0,
            'distance_place_rate': 0.0,
            'distance_avg_finish': 5.0,
            'distance_experience': 0.0,
            'distance_recent_form': 5.0
        }
    
    # Calculate distance-specific metrics
    distance_starts = len(distance_races)
    distance_wins = distance_races['won'].sum()
    distance_places = distance_races['placed'].sum()
    distance_win_rate = distance_wins / distance_starts
    distance_place_rate = distance_places / distance_starts
    distance_avg_finish = distance_races['finish_position'].mean()
    distance_experience = np.log1p(distance_starts)
    
    # Recent form at distance (last 3 races or all if fewer)
    recent_distance_races = distance_races.tail(3)
    distance_recent_form = recent_distance_races['finish_position'].mean()
    
    return {
        'distance_starts': distance_starts,
        'distance_win_rate': distance_win_rate,
        'distance_place_rate': distance_place_rate,
        'distance_avg_finish': distance_avg_finish,
        'distance_experience': distance_experience,
        'distance_recent_form': distance_recent_form
    }

def calculate_track_condition_performance(horse_name: str, target_condition: str, training_df: pd.DataFrame) -> dict:
    """Calculate track condition-specific performance metrics for a horse."""
    
    # Filter for this horse's races on the target track condition
    horse_races = training_df[training_df['horse_name'] == horse_name]
    condition_races = horse_races[horse_races['track_condition'].str.lower() == target_condition.lower()]
    
    if len(condition_races) == 0:
        # No experience on this track condition
        return {
            'condition_starts': 0,
            'condition_win_rate': 0.0,
            'condition_place_rate': 0.0,
            'condition_avg_finish': 5.0,
            'condition_experience': 0.0,
            'condition_recent_form': 5.0
        }
    
    # Calculate track condition-specific metrics
    condition_starts = len(condition_races)
    condition_wins = condition_races['won'].sum()
    condition_places = condition_races['placed'].sum()
    condition_win_rate = condition_wins / condition_starts
    condition_place_rate = condition_places / condition_starts
    condition_avg_finish = condition_races['finish_position'].mean()
    condition_experience = np.log1p(condition_starts)
    
    # Recent form on track condition (last 3 races or all if fewer)
    recent_condition_races = condition_races.tail(3)
    condition_recent_form = recent_condition_races['finish_position'].mean()
    
    return {
        'condition_starts': condition_starts,
        'condition_win_rate': condition_win_rate,
        'condition_place_rate': condition_place_rate,
        'condition_avg_finish': condition_avg_finish,
        'condition_experience': condition_experience,
        'condition_recent_form': condition_recent_form
    }

def create_upcoming_race_data(target_horses: list, race_info: dict, training_df: pd.DataFrame, horse_race_details: dict = None) -> pd.DataFrame:
    """Create upcoming race data with horse-specific historical features (fully deterministic)."""
    
    # Set all seeds for complete reproducibility
    np.random.seed(42)
    
    # Sort the DataFrame to ensure consistent ordering
    training_df_sorted = training_df.sort_values(['horse_name', 'race_date']).reset_index(drop=True)
    
    # Get horse-specific features from training data with explicit sorting
    horse_profiles = training_df_sorted.groupby('horse_name', sort=True).agg({
        'standard_time_diff_overall': 'mean',
        'standard_time_diff_800m': 'mean', 
        'standard_time_diff_400m': 'mean',
        'standard_time_diff_final': 'mean',
        'career_win_rate_to_date': 'last',  # Most recent
        'career_starts_to_date': 'last',
        'recent_form_3_races': 'last',
        'won': 'mean',
        'placed': 'mean',
        'finish_position': 'mean'
    }).round(3)
    
    race_data = []
    
    # Process horses in sorted order for consistency
    for i, horse in enumerate(sorted(target_horses)):
        # Get horse-specific data if available
        if horse in horse_profiles.index:
            profile = horse_profiles.loc[horse]
            
            # Calculate venue performance from historical data
            target_venue = race_info.get('venue', 'Unknown')
            venue_performance = calculate_venue_performance(horse, target_venue, training_df_sorted)
            
            # Calculate distance performance from historical data
            distance_str = str(race_info.get('distance', 1400))
            target_distance = int(distance_str.replace('m', '').replace('metres', '').strip())
            distance_performance = calculate_distance_performance(horse, target_distance, training_df_sorted)
            
            # Calculate track condition performance from historical data
            target_condition = race_info.get('track_condition', 'Good')
            condition_performance = calculate_track_condition_performance(horse, target_condition, training_df_sorted)
            
            # Use ACTUAL race details if available, otherwise fall back to reasonable defaults
            if horse_race_details and horse in horse_race_details:
                horse_details = horse_race_details[horse]
                barrier_position = horse_details['barrier']
                weight = horse_details['weight']
                saddlecloth_number = horse_details['saddlecloth_number']
            else:
                # Fallback: Use barrier assignment based on sorted order
                barrier_position = (i % 16) + 1  # Spread across 16 barriers
                weight = 57.5 + (i * 0.1)  # Small weight increments
                saddlecloth_number = i + 1  # Sequential numbering
            
            race_record = {
                'horse_name': horse,
                'barrier': barrier_position,
                'weight': weight,
                'saddlecloth_number': saddlecloth_number,
                'field_size': len(target_horses),
                
                # Use horse's historical performance (deterministic)
                'standard_time_diff_overall': float(profile['standard_time_diff_overall']) if pd.notna(profile['standard_time_diff_overall']) else 0.0,
                'standard_time_diff_800m': float(profile['standard_time_diff_800m']) if pd.notna(profile['standard_time_diff_800m']) else 0.0,
                'standard_time_diff_400m': float(profile['standard_time_diff_400m']) if pd.notna(profile['standard_time_diff_400m']) else 0.0,
                'standard_time_diff_final': float(profile['standard_time_diff_final']) if pd.notna(profile['standard_time_diff_final']) else 0.0,
                'career_win_rate_to_date': float(profile['career_win_rate_to_date']) if pd.notna(profile['career_win_rate_to_date']) else 0.1,
                'career_starts_to_date': float(profile['career_starts_to_date']) if pd.notna(profile['career_starts_to_date']) else 5.0,
                'recent_form_3_races': float(profile['recent_form_3_races']) if pd.notna(profile['recent_form_3_races']) else 5.0,
                
                # Race context (calculated from actual performance history)
                'venue_win_rate': venue_performance['venue_win_rate'],
                'venue_place_rate': venue_performance['venue_place_rate'],
                'venue_avg_finish': venue_performance['venue_avg_finish'],
                'venue_experience': venue_performance['venue_experience'],
                'venue_recent_form': venue_performance['venue_recent_form'],
                'distance_win_rate': distance_performance['distance_win_rate'],
                'distance_place_rate': distance_performance['distance_place_rate'],
                'distance_avg_finish': distance_performance['distance_avg_finish'],
                'distance_experience': distance_performance['distance_experience'],
                'distance_recent_form': distance_performance['distance_recent_form'],
                'condition_win_rate': condition_performance['condition_win_rate'],
                'condition_place_rate': condition_performance['condition_place_rate'],
                'condition_avg_finish': condition_performance['condition_avg_finish'],
                'condition_experience': condition_performance['condition_experience'],
                'condition_recent_form': condition_performance['condition_recent_form'],
                'same_class': True,
                'days_to_target_race': 0,
                'distance_difference': 0,
                'venue': race_info.get('venue', 'Unknown'),
                'track_condition': race_info.get('track_condition', 'Good'),
                
                # Derived features (calculated from actual race conditions)
                'recency_score': 1.0,  # Current race = maximum recency
                'experience_factor': np.log1p(float(profile['career_starts_to_date']) if pd.notna(profile['career_starts_to_date']) else 5.0) / 4.0,
                'recent_form_score': 10.0 - min(10, max(1, float(profile['recent_form_3_races']) if pd.notna(profile['recent_form_3_races']) else 5.0)),
                'overall_win_rate': float(profile['won']) if pd.notna(profile['won']) else 0.1,
                'overall_place_rate': float(profile['placed']) if pd.notna(profile['placed']) else 0.3,
                'avg_finish_position': float(profile['finish_position']) if pd.notna(profile['finish_position']) else 5.0,
                'distance_change_penalty': calculate_distance_penalty(profile, race_info.get('distance', 1400), training_df_sorted, horse),
                'barrier_disadvantage': abs(barrier_position - 8) / 8.0,  # Based on actual barrier
                'weight_burden': max(0, weight - 57.0) / 5.0,  # Based on actual weight
                'field_difficulty': max(0, (len(target_horses) - 8) / 8.0)  # Field size relative to 8-horse baseline
            }
        else:
            # Default values for horses not in training data
            # Calculate performance metrics (will return zeros for horses not in training data)
            target_venue = race_info.get('venue', 'Unknown')
            venue_performance = calculate_venue_performance(horse, target_venue, training_df_sorted)
            
            distance_str = str(race_info.get('distance', 1400))
            target_distance = int(distance_str.replace('m', '').replace('metres', '').strip())
            distance_performance = calculate_distance_performance(horse, target_distance, training_df_sorted)
            
            target_condition = race_info.get('track_condition', 'Good')
            condition_performance = calculate_track_condition_performance(horse, target_condition, training_df_sorted)
            
            # Use ACTUAL race details if available, otherwise fall back to defaults
            if horse_race_details and horse in horse_race_details:
                horse_details = horse_race_details[horse]
                barrier_position = horse_details['barrier']
                weight = horse_details['weight']
                saddlecloth_number = horse_details['saddlecloth_number']
            else:
                # Fallback: Use barrier assignment based on sorted order
                barrier_position = (i % 16) + 1  # Spread across 16 barriers
                weight = 57.5 + (i * 0.1)  # Small weight increments
                saddlecloth_number = i + 1  # Sequential numbering
            
            race_record = {
                'horse_name': horse,
                'barrier': barrier_position,
                'weight': weight,
                'saddlecloth_number': saddlecloth_number,
                'field_size': len(target_horses),
                'standard_time_diff_overall': 0.0,
                'standard_time_diff_800m': 0.0,
                'standard_time_diff_400m': 0.0,
                'standard_time_diff_final': 0.0,
                'career_win_rate_to_date': 0.1,
                'career_starts_to_date': 5.0,
                'recent_form_3_races': 5.0,
                'venue_win_rate': venue_performance['venue_win_rate'],
                'venue_place_rate': venue_performance['venue_place_rate'],
                'venue_avg_finish': venue_performance['venue_avg_finish'],
                'venue_experience': venue_performance['venue_experience'],
                'venue_recent_form': venue_performance['venue_recent_form'],
                'distance_win_rate': distance_performance['distance_win_rate'],
                'distance_place_rate': distance_performance['distance_place_rate'],
                'distance_avg_finish': distance_performance['distance_avg_finish'],
                'distance_experience': distance_performance['distance_experience'],
                'distance_recent_form': distance_performance['distance_recent_form'],
                'condition_win_rate': condition_performance['condition_win_rate'],
                'condition_place_rate': condition_performance['condition_place_rate'],
                'condition_avg_finish': condition_performance['condition_avg_finish'],
                'condition_experience': condition_performance['condition_experience'],
                'condition_recent_form': condition_performance['condition_recent_form'],
                'same_class': True,
                'days_to_target_race': 0,
                'distance_difference': 0,
                'venue': race_info.get('venue', 'Unknown'),
                'track_condition': race_info.get('track_condition', 'Good'),
                'recency_score': 1.0,
                'experience_factor': 0.4,
                'recent_form_score': 5.0,
                'overall_win_rate': 0.1,
                'overall_place_rate': 0.3,
                'avg_finish_position': 5.0,
                'distance_change_penalty': 0.0,
                'barrier_disadvantage': abs(barrier_position - 8) / 8.0,  # Based on actual barrier
                'weight_burden': max(0, weight - 57.0) / 5.0,  # Based on actual weight
                'field_difficulty': max(0, (len(target_horses) - 8) / 8.0)  # Field size effect
            }
        
        race_data.append(race_record)
    
    # Create DataFrame and sort by horse name for consistency
    df = pd.DataFrame(race_data)
    df = df.sort_values('horse_name').reset_index(drop=True)
    
    return df

def main():
    """Main function to run neural racing predictor."""
    parser = argparse.ArgumentParser(description='Neural Network Horse Racing Predictor')
    parser.add_argument('date', help='Race date (YYYY-MM-DD)')
    parser.add_argument('venue', help='Venue name')
    parser.add_argument('race_number', type=int, help='Race number')
    parser.add_argument('--api-key', help='Racing.com API key (optional if RACING_API_KEY in .env)')
    parser.add_argument('--output-dir', default='neural_predictions', help='Output directory')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden-dims', nargs='+', type=int, default=[128, 64, 32], 
                       help='Hidden layer dimensions')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'], help='Training device')
    parser.add_argument('--save-data', action='store_true', help='Save training data')
    parser.add_argument('--store-cosmos', action='store_true', default=True, 
                       help='Store results in Cosmos DB (default: True)')
    parser.add_argument('--no-cosmos', action='store_true',
                       help='Disable Cosmos DB storage')
    
    args = parser.parse_args()
    
    # Get API key from args or environment
    api_key = args.api_key or os.getenv('RACING_API_KEY')
    if not api_key:
        logger.error("API key required. Either pass --api-key or set RACING_API_KEY in .env file")
        sys.exit(1)
    
    print("🧠 Neural Network Horse Racing Predictor")
    print("="*60)
    print(f"Target Race: {args.date} {args.venue} Race {args.race_number}")
    print(f"Device: {args.device}")
    print(f"Model: Neural network with layers {args.hidden_dims}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Step 1: Extract Training Data
        print(f"\n📊 STEP 1: Extracting Training Data")
        print("-" * 40)
        
        # Initialize API client and extractor
        client = RacingComAPIClient(api_key)
        extractor = BayesianTrainingDataExtractor(client)
        
        # Define target race
        target = RaceTarget(
            date=args.date,
            venue=args.venue,
            race_number=args.race_number
        )
        
        # Extract training records
        logger.info("Extracting historical training data...")
        records = extractor.extract_training_records(target)
        
        if not records:
            logger.error("No training records found. Check race details.")
            sys.exit(1)
        
        # Convert to DataFrame
        df = extractor.records_to_dataframe(records)
        
        # Get list of horses in target race
        target_horses = df['horse_name'].unique().tolist()
        
        print(f"✅ Extracted {len(records)} historical records for {len(target_horses)} horses")
        
        # Save training data if requested
        if args.save_data:
            data_filename = f"neural_training_data_{args.date}_{args.venue.replace(' ', '_')}_race{args.race_number}.csv"
            data_path = os.path.join(args.output_dir, data_filename)
            df.to_csv(data_path, index=False)
            logger.info(f"Training data saved to: {data_path}")
        
        # Print training summary
        print_training_summary(df, target_horses)
        
        # Step 2: Train Neural Network Model
        print(f"\n🧠 STEP 2: Training Neural Network Model")
        print("-" * 40)
        
        # Check for GPU availability
        if args.device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Using CPU.")
            args.device = 'cpu'
        
        # Initialize model
        model = NeuralRacingModel(
            hidden_dims=args.hidden_dims,
            learning_rate=args.learning_rate,
            device=args.device
        )
        
        logger.info(f"Training neural network with {args.epochs} epochs...")
        metrics = model.train(
            df, 
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_split=0.2
        )
        
        print(f"✅ Neural network training completed")
        
        # Step 3: Generate Predictions
        print(f"\n🎯 STEP 3: Generating Race Predictions")
        print("-" * 40)
        
        # Extract ACTUAL race information from API
        logger.info("Fetching actual race details from API...")
        try:
            # Find the race and get its details
            meet_code, actual_race_number = extractor.find_race(target)
            race_form_data = client.get_race_entries(meet_code, actual_race_number)
            
            # Extract race details
            race_details = race_form_data  # get_race_entries returns the race form directly
            race_info = {
                'date': args.date,
                'venue': args.venue,
                'race_number': args.race_number,
                'distance': race_details.get('distance', 1400),  # Actual distance
                'track_condition': race_details.get('trackCondition', 'Good'),  # Actual condition
                'class': race_details.get('class') or race_details.get('rdcClass') or race_details.get('group') or 'Unknown',  # Actual class
                'meet_code': meet_code,
                'actual_race_number': actual_race_number
            }
            
            # Get actual horse entries with barriers and weights
            race_entries = race_details.get('formRaceEntries', [])
            horse_race_details = {}
            
            for entry in race_entries:
                horse_name = entry.get('horseName', '')
                if horse_name in target_horses:
                    # Parse weight (remove 'kg' suffix if present)
                    weight_str = str(entry.get('weight', '57.5'))
                    weight = float(weight_str.replace('kg', '').strip()) if weight_str else 57.5
                    
                    horse_race_details[horse_name] = {
                        'barrier': int(entry.get('barrierNumber', entry.get('barrier', 8))),  # Actual barrier
                        'weight': weight,  # Actual weight (parsed from "62kg" format)
                        'saddlecloth_number': int(entry.get('raceEntryNumber', 1)),  # Actual race entry number
                        'jockey': entry.get('jockeyName', 'Unknown'),
                        'trainer': entry.get('trainerName', 'Unknown')
                    }
            
            logger.info(f"✅ Extracted race details: {race_info['distance']}m, {race_info['track_condition']} track")
            logger.info(f"✅ Found race entries for {len(horse_race_details)} horses")
            
        except Exception as e:
            logger.warning(f"Could not fetch race details from API: {e}")
            logger.info("Using fallback race information...")
            
            # Fallback to defaults if API fails
            race_info = {
                'date': args.date,
                'venue': args.venue,
                'race_number': args.race_number,
                'distance': 1400,
                'track_condition': 'Good',
                'class': 'Unknown'
            }
            horse_race_details = {}
        
        print_race_summary(race_info, len(target_horses))
        
        # Create upcoming race data with ACTUAL race details
        upcoming_race_data = create_upcoming_race_data(target_horses, race_info, df, horse_race_details)
        
        # Generate predictions
        predictions = model.predict_race(upcoming_race_data)
        
        # Add saddlecloth numbers to predictions DataFrame
        saddlecloth_lookup = dict(zip(upcoming_race_data['horse_name'], upcoming_race_data['saddlecloth_number']))
        predictions['saddlecloth_number'] = predictions['horse_name'].map(saddlecloth_lookup)
        
        print(f"✅ Predictions generated for {len(predictions)} horses")
        
        # Print results
        print_model_results(metrics, predictions, race_info)
        
        # Show feature importance
        try:
            importance_df = model.get_feature_importance()
            if len(importance_df) > 0:
                print(f"\n🎯 Top 10 Most Important Features:")
                for _, row in importance_df.head(10).iterrows():
                    print(f"  {row['feature']}: {row['importance']:.3f}")
        except Exception as e:
            logger.warning(f"Could not calculate feature importance: {e}")
        
        # Step 4: Save Results
        print(f"\n💾 STEP 4: Saving Results")
        print("-" * 40)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save predictions
        pred_filename = f"neural_predictions_{args.date}_{args.venue.replace(' ', '_')}_race{args.race_number}_{timestamp}.csv"
        pred_path = os.path.join(args.output_dir, pred_filename)
        predictions.to_csv(pred_path, index=False)
        
        # Save detailed results
        results_filename = f"neural_results_{args.date}_{args.venue.replace(' ', '_')}_race{args.race_number}_{timestamp}.json"
        results_path = os.path.join(args.output_dir, results_filename)
        
        results_data = {
            'race_info': race_info,
            'model_metrics': metrics,
            'model_config': {
                'hidden_dims': args.hidden_dims,
                'learning_rate': args.learning_rate,
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'device': args.device
            },
            'predictions': predictions.to_dict('records'),
            'feature_importance': importance_df.to_dict('records') if 'importance_df' in locals() else [],
            'training_summary': {
                'total_records': len(df),
                'horses': len(target_horses),
                'date_range': [df['race_date'].min(), df['race_date'].max()],
                'win_rate': df['won'].mean(),
                'place_rate': df['placed'].mean()
            }
        }
        
# Store results in Cosmos DB (if enabled and available)
        store_in_cosmos = (args.store_cosmos and not args.no_cosmos and COSMOS_DB_AVAILABLE)
        cosmos_success = False
        prediction_id = f"neural_{args.date}_{args.venue.replace(' ', '_')}_race{args.race_number}_{timestamp}"
        
        if store_in_cosmos:
            try:
                cosmos_client = CosmosDBClient()
                
                # Prepare document for Cosmos DB
                cosmos_document = {
                    'id': prediction_id,
                    'race_date': args.date,
                    'venue': args.venue,
                    'race_number': args.race_number,
                    'prediction_type': 'neural_network',
                    'created_at': datetime.now().isoformat(),
                    'model_type': 'neural_racing_predictor',
                    **results_data  # Include all the existing results data
                }
                
                # Store in Cosmos DB with separate prediction_id and prediction_data
                cosmos_success = cosmos_client.store_prediction(prediction_id, cosmos_document)
                if cosmos_success:
                    print(f"✅ Results stored in Cosmos DB with ID: {prediction_id}")
                else:
                    print(f"❌ Failed to store results in Cosmos DB")
                
            except Exception as e:
                print(f"⚠️  Warning: Could not store results in Cosmos DB: {e}")
                logger.warning(f"Cosmos DB storage failed: {e}")
        elif args.no_cosmos:
            print("ℹ️  Cosmos DB storage disabled by --no-cosmos flag")
        elif not COSMOS_DB_AVAILABLE:
            print("ℹ️  Cosmos DB not available - install cosmos_db_client.py for cloud storage")
        
        # Only save local files if Cosmos DB storage failed or was not attempted
        if not cosmos_success:
            print(f"💾 Saving results locally...")
            
            # Save predictions CSV
            predictions.to_csv(pred_path, index=False)
            
            # Save detailed results JSON
            with open(results_path, 'w') as f:
                json.dump(results_data, f, indent=2, default=str)
        else:
            print(f"ℹ️  Skipping local file storage - results stored in Cosmos DB")
        
        print(f"✅ Results saved:")
        if not cosmos_success:
            print(f"  Local CSV: {pred_path}")
            print(f"  Local JSON: {results_path}")
        else:
            print(f"  Local files: ❌ Skipped (stored in Cosmos DB)")
            
        if cosmos_success:
            print(f"  Cosmos DB: ✅ Stored with ID: {prediction_id}")
        elif store_in_cosmos:
            print(f"  Cosmos DB: ❌ Storage failed")
        else:
            print(f"  Cosmos DB: ❌ Not attempted (disabled or unavailable)")
        
        # Final summary
        print(f"\n🏆 PREDICTION SUMMARY")
        print("="*60)
        
        top_pick = predictions.iloc[0]
        top_pick_number = top_pick.get('saddlecloth_number', top_pick.get('horse_number', 1))
        print(f"🥇 Top Selection: #{top_pick_number} {top_pick['horse_name']}")
        print(f"   Win Probability: {top_pick['win_probability']:.1%}")
        print(f"   Place Probability: {top_pick['place_probability']:.1%}")
        print(f"   Expected Finish: {top_pick['predicted_finish']:.1f}")
        print(f"   Confidence: {top_pick['confidence']:.2f}")
        
        print(f"\n✅ Neural network prediction pipeline completed!")
        print(f"📁 All results saved to: {args.output_dir}")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if metrics['val_win_auc'] > 0.65:
            print(f"  ✅ Model shows good predictive power")
            print(f"  🎯 Top 3 selections are worth considering")
        elif metrics['val_win_auc'] > 0.55:
            print(f"  ⚠️  Model shows moderate predictive power")
            print(f"  🔍 Focus on top selection and consider other factors")
        else:
            print(f"  ❌ Model shows poor predictive power")
            print(f"  🔍 Consider gathering more training data")
        
        if len(df) > 300:
            print(f"  ✅ Good amount of training data ({len(df)} records)")
        else:
            print(f"  ⚠️  Limited training data ({len(df)} records)")
            print(f"  🔍 More historical data would improve reliability")
        
        # Neural network specific advice
        print(f"\n🧠 Neural Network Advantages Used:")
        print(f"  • Multi-task learning (win, place, finish position)")
        print(f"  • Works without head-to-head race history")
        print(f"  • Uses standardized performance metrics")
        print(f"  • Handles complex feature interactions")
        print(f"  • Provides calibrated probabilities")
        
    except Exception as e:
        logger.error(f"Error in neural prediction pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()