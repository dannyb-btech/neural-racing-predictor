#!/usr/bin/env python3
"""
Simple test API for Neural Racing Predictor.
"""

import os
import json
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
import pandas as pd
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from cosmos_db_client import CosmosDBClient

# Import neural racing predictor components
from racing_com_api_client import RacingComAPIClient
from feature_creation import BayesianTrainingDataExtractor, RaceTarget
from neural_racing_model import NeuralRacingModel
from neural_racing_predictor import (
    print_race_summary, print_training_summary, print_model_results,
    create_upcoming_race_data
)

# Initialize FastAPI app
app = FastAPI(
    title="Neural Racing Predictor API - Test",
    description="Simple test API for neural racing predictions",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple request/response models
class PredictionRequest(BaseModel):
    date: str
    venue: str
    race_number: int
    use_neural_network: Optional[bool] = False  # Default to test predictions

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    cosmos_db: str

# Initialize Cosmos DB client on startup
cosmos_client = None

def generate_neural_prediction(date: str, venue: str, race_number: int, api_key: str) -> Dict[str, Any]:
    """Generate real neural network predictions."""
    try:
        # Step 1: Extract Training Data
        print(f"📊 Extracting training data for {venue} Race {race_number} on {date}")
        
        # Initialize API client and extractor
        client = RacingComAPIClient(api_key)
        extractor = BayesianTrainingDataExtractor(client)
        
        # Define target race
        target = RaceTarget(
            date=date,
            venue=venue,
            race_number=race_number
        )
        
        # Extract training records
        training_records = extractor.extract_training_records(target)
        
        if len(training_records) < 50:
            raise ValueError(f"Insufficient training data: {len(training_records)} records. Need at least 50.")
        
        # Convert to DataFrame
        df = pd.DataFrame([record.__dict__ for record in training_records])
        target_horses = list(df['horse_name'].unique())
        
        print(f"✅ Extracted {len(training_records)} training records for {len(target_horses)} horses")
        
        # Step 2: Train Neural Network
        print(f"🧠 Training neural network...")
        
        model = NeuralRacingModel(
            hidden_dims=[128, 64, 32],
            learning_rate=0.001,
            device='cpu'
        )
        
        metrics = model.train(
            df, 
            epochs=50,  # Reduced epochs for API speed
            batch_size=32,
            validation_split=0.2
        )
        
        print(f"✅ Neural network training completed")
        
        # Step 3: Generate Predictions
        print(f"🎯 Generating race predictions...")
        
        # Extract race information from API
        race_info = {
            'date': date,
            'venue': venue,
            'race_number': race_number,
            'distance': 1400,
            'track_condition': 'Good',
            'class': 'Unknown'
        }
        
        horse_race_details = {}
        
        try:
            # Try to get actual race details
            meet_code, actual_race_number = extractor.find_race(target)
            race_form_data = client.get_race_entries(meet_code, actual_race_number)
            
            # Extract race details
            race_info.update({
                'distance': race_form_data.get('distance', 1400),
                'track_condition': race_form_data.get('trackCondition', 'Good'),
                'class': (race_form_data.get('class') or 
                         race_form_data.get('rdcClass') or 
                         race_form_data.get('group') or 'Unknown'),
                'meet_code': meet_code,
                'actual_race_number': actual_race_number
            })
            
            # Get race entries
            race_entries = race_form_data.get('formRaceEntries', [])
            for entry in race_entries:
                horse_name = entry.get('horseName', '')
                if horse_name in target_horses:
                    weight_str = str(entry.get('weight', '57.5'))
                    weight = float(weight_str.replace('kg', '').strip()) if weight_str else 57.5
                    
                    horse_race_details[horse_name] = {
                        'barrier': int(entry.get('barrierNumber', entry.get('barrier', 8))),
                        'weight': weight,
                        'saddlecloth_number': int(entry.get('raceEntryNumber', 1)),
                        'jockey': entry.get('jockeyName', 'Unknown'),
                        'trainer': entry.get('trainerName', 'Unknown')
                    }
            
            print(f"✅ Extracted race details from API")
            
        except Exception as e:
            print(f"⚠️ Could not fetch race details from API: {e}")
            print("Using fallback race information...")
        
        # Create upcoming race data
        upcoming_race_data = create_upcoming_race_data(target_horses, race_info, df, horse_race_details)
        
        # Generate predictions
        predictions = model.predict_race(upcoming_race_data)
        
        # Convert predictions to serializable format
        predictions_list = []
        for _, pred in predictions.iterrows():
            predictions_list.append({
                'rank': int(pred.get('rank', 0)),
                'horse_name': pred['horse_name'],
                'barrier': int(pred.get('barrier', 0)) if pd.notna(pred.get('barrier')) else None,
                'weight': float(pred.get('weight', 0)) if pd.notna(pred.get('weight')) else None,
                'jockey': pred.get('jockey', 'Unknown'),
                'trainer': pred.get('trainer', 'Unknown'), 
                'win_probability': float(pred['win_probability']),
                'place_probability': float(pred['place_probability']),
                'predicted_finish': float(pred['predicted_finish']),
                'confidence': float(pred.get('confidence', 0.8)),
                'saddlecloth_number': int(pred.get('saddlecloth_number', 0)) if pd.notna(pred.get('saddlecloth_number')) else None
            })
        
        # Sort by predicted finish position
        predictions_list.sort(key=lambda x: x['predicted_finish'])
        
        # Add ranks
        for i, pred in enumerate(predictions_list):
            pred['rank'] = i + 1
        
        print(f"✅ Generated predictions for {len(predictions_list)} horses")
        
        return {
            'race_info': race_info,
            'predictions': predictions_list,
            'model_metrics': {
                'training_records': len(training_records),
                'field_size': len(target_horses),
                'win_auc': float(metrics.get('win_auc', 0.0)),
                'place_auc': float(metrics.get('place_auc', 0.0)),
                'win_loss': float(metrics.get('win_loss', 0.0)),
                'place_loss': float(metrics.get('place_loss', 0.0))
            },
            'created_at': datetime.now(timezone.utc).isoformat(),
            'note': 'Neural network prediction using historical racing data'
        }
        
    except Exception as e:
        raise Exception(f"Neural prediction failed: {str(e)}")

@app.on_event("startup")
async def startup_event():
    global cosmos_client
    try:
        cosmos_client = CosmosDBClient()
        print("✅ Cosmos DB client initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize Cosmos DB: {e}")

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint."""
    cosmos_status = "connected" if cosmos_client else "disconnected"
    return HealthResponse(
        status="ok",
        timestamp=datetime.now(timezone.utc).isoformat(),
        cosmos_db=cosmos_status
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check."""
    cosmos_status = "disconnected"
    if cosmos_client:
        try:
            # Test cosmos connection with a simple query
            stats = cosmos_client.get_database_stats()
            cosmos_status = "connected"
        except:
            cosmos_status = "error"
    
    return HealthResponse(
        status="ok",
        timestamp=datetime.now(timezone.utc).isoformat(),
        cosmos_db=cosmos_status
    )

@app.post("/predict")
async def create_prediction(request: PredictionRequest):
    """Create a prediction - either test or real neural network."""
    if not cosmos_client:
        raise HTTPException(status_code=500, detail="Cosmos DB not initialized")
    
    # Get API key from environment
    api_key = os.getenv('RACING_API_KEY')
    if not api_key:
        raise HTTPException(status_code=500, detail="Racing API key not configured")
    
    # Generate prediction ID
    prediction_id = str(uuid.uuid4())
    
    try:
        if request.use_neural_network:
            # Generate real neural network prediction
            logger.info(f"Generating neural network prediction for {request.venue} Race {request.race_number}")
            prediction_data = generate_neural_prediction(
                request.date, 
                request.venue, 
                request.race_number, 
                api_key
            )
            prediction_data['prediction_id'] = prediction_id
            
        else:
            # Generate test prediction (existing logic)
            prediction_data = {
                'prediction_id': prediction_id,
                'race_info': {
                    'date': request.date,
                    'venue': request.venue,
                    'race_number': request.race_number,
                    'distance': 1600,
                    'track_condition': 'Good'
                },
                'predictions': [
                    {
                        'rank': 1,
                        'horse_name': 'Test Horse 1',
                        'barrier': 3,
                        'weight': 58.5,
                        'jockey': 'Test Jockey A',
                        'trainer': 'Test Trainer A',
                        'win_probability': 0.25,
                        'place_probability': 0.65,
                        'predicted_finish': 2.1,
                        'confidence': 0.85,
                        'saddlecloth_number': 1
                    },
                    {
                        'rank': 2,
                        'horse_name': 'Test Horse 2',
                        'barrier': 8,
                        'weight': 57.0,
                        'jockey': 'Test Jockey B',
                        'trainer': 'Test Trainer B',
                        'win_probability': 0.20,
                        'place_probability': 0.60,
                        'predicted_finish': 3.2,
                        'confidence': 0.80,
                        'saddlecloth_number': 2
                    },
                    {
                        'rank': 3,
                        'horse_name': 'Test Horse 3',
                        'barrier': 5,
                        'weight': 56.5,
                        'jockey': 'Test Jockey C',
                        'trainer': 'Test Trainer C',
                        'win_probability': 0.18,
                        'place_probability': 0.55,
                        'predicted_finish': 4.1,
                        'confidence': 0.75,
                        'saddlecloth_number': 3
                    },
                    {
                        'rank': 4,
                        'horse_name': 'Test Horse 4',
                        'barrier': 1,
                        'weight': 59.0,
                        'jockey': 'Test Jockey D',
                        'trainer': 'Test Trainer D',
                        'win_probability': 0.15,
                        'place_probability': 0.50,
                        'predicted_finish': 5.2,
                        'confidence': 0.70,
                        'saddlecloth_number': 4
                    }
                ],
                'model_metrics': {
                    'training_records': 1000,
                    'field_size': 8,
                    'win_auc': 0.75,
                    'place_auc': 0.82
                },
                'created_at': datetime.now(timezone.utc).isoformat(),
                'note': 'This is a test prediction - not real neural network output'
            }
            
    except Exception as e:
        logger.error(f"Prediction generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    # Store in Cosmos DB
    success = cosmos_client.store_prediction(prediction_id, prediction_data)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to store prediction")
    
    return {
        'prediction_id': prediction_id,
        'status': 'completed',
        'message': 'Test prediction created successfully',
        'data': prediction_data
    }

@app.get("/predictions/search")
async def search_predictions(date: Optional[str] = None, venue: Optional[str] = None):
    """Search predictions."""
    if not cosmos_client:
        raise HTTPException(status_code=500, detail="Cosmos DB not initialized")
    
    predictions = cosmos_client.search_predictions(date=date, venue=venue, limit=10)
    return {
        'predictions': predictions,
        'total': len(predictions)
    }

@app.get("/predictions/{prediction_id}")
async def get_prediction(prediction_id: str):
    """Retrieve a prediction by ID."""
    if not cosmos_client:
        raise HTTPException(status_code=500, detail="Cosmos DB not initialized")
    
    prediction = cosmos_client.get_prediction(prediction_id)
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    
    return prediction

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)