#!/usr/bin/env python3
"""
FastAPI application for Neural Racing Predictor API.

Provides RESTful endpoints for generating and retrieving horse racing predictions.
"""

import os
import json
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
import logging

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Import our existing neural racing predictor components
from racing_com_api_client import RacingComAPIClient
from feature_creation import BayesianTrainingDataExtractor, RaceTarget
from neural_racing_model import NeuralRacingModel
from neural_racing_predictor import (
    print_race_summary, print_training_summary, print_model_results,
    create_upcoming_race_data
)
from cosmos_db_client import CosmosDBClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Neural Racing Predictor API",
    description="API for generating AI-powered horse racing predictions",
    version="1.0.0"
)

# Configure CORS for web app access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API requests/responses
class PredictionRequest(BaseModel):
    """Request model for generating race predictions."""
    date: str = Field(..., description="Race date in YYYY-MM-DD format", example="2025-07-25")
    venue: str = Field(..., description="Race venue name", example="Flemington")
    race_number: int = Field(..., description="Race number", example=7, ge=1, le=12)
    api_key: str = Field(..., description="Racing.com API key")

class PredictionResponse(BaseModel):
    """Response model for prediction requests."""
    prediction_id: str = Field(..., description="Unique identifier for the prediction")
    status: str = Field(..., description="Status of the prediction", example="completed")
    message: str = Field(..., description="Status message")
    race_info: Optional[Dict[str, Any]] = None
    predictions: Optional[List[Dict[str, Any]]] = None
    model_metrics: Optional[Dict[str, Any]] = None
    created_at: str = Field(..., description="ISO timestamp when prediction was created")

class PredictionSearchResponse(BaseModel):
    """Response model for prediction search."""
    predictions: List[Dict[str, Any]] = Field(..., description="List of matching predictions")
    total_count: int = Field(..., description="Total number of matching predictions")

# Initialize Cosmos DB client
try:
    cosmos_client = CosmosDBClient()
    logger.info("Cosmos DB client initialized successfully")
except Exception as e:
    logger.warning(f"Cosmos DB initialization failed: {e}. Using in-memory storage.")
    cosmos_client = None

# Fallback in-memory storage
prediction_store: Dict[str, Dict[str, Any]] = {}

def generate_prediction_core(date: str, venue: str, race_number: int, api_key: str) -> Dict[str, Any]:
    """
    Core prediction logic extracted from neural_racing_predictor.py main function.
    
    Returns a dictionary containing all prediction data and metadata.
    """
    try:
        # Set up API client
        client = RacingComAPIClient(api_key)
        
        # Create race target
        target = RaceTarget(date=date, venue=venue, race_number=race_number)
        
        # Extract training data
        extractor = BayesianTrainingDataExtractor(client)
        training_records = extractor.extract_training_records(target)
        
        if len(training_records) < 50:
            raise ValueError(f"Insufficient training data: {len(training_records)} records (minimum 50 required)")
        
        # Convert to DataFrame
        training_df = extractor.records_to_dataframe(training_records)
        
        # Get target race horses
        meet_code, actual_race_number = extractor.find_race(target)
        race_form_data = client.get_race_entries(meet_code, actual_race_number)
        
        # Extract race details
        race_details = race_form_data
        race_info = {
            'date': date,
            'venue': venue,
            'race_number': race_number,
            'distance': race_details.get('distance', 1400),
            'track_condition': race_details.get('trackCondition', 'Good'),  
            'class': race_details.get('class') or race_details.get('rdcClass') or race_details.get('group') or 'Unknown',
            'meet_code': meet_code,
            'actual_race_number': actual_race_number
        }
        
        # Get horse entries
        race_entries = race_details.get('formRaceEntries', [])
        target_horses = []
        horse_race_details = {}
        
        for entry in race_entries:
            if entry.get('scratched', False):
                continue
                
            horse_name = entry.get('horseName')
            target_horses.append({
                'horse_name': horse_name,
                'horse_code': entry.get('horseCode'),
                'barrier': entry.get('barrierNumber'),
                'weight': entry.get('weight'),
                'jockey': entry.get('jockeyName'),
                'trainer': entry.get('trainerName')
            })
            
            horse_race_details[horse_name] = {
                'barrier': entry.get('barrierNumber'),
                'weight': entry.get('weight'),
                'jockey': entry.get('jockeyName'),
                'trainer': entry.get('trainerName')
            }
        
        if not target_horses:
            raise ValueError("No active horses found in target race")
        
        # Create upcoming race data
        upcoming_race_df = create_upcoming_race_data(target_horses, race_info, training_df, horse_race_details)
        
        # Train model and make predictions
        model = NeuralRacingModel()
        metrics = model.train(training_df)
        predictions = model.predict_race(upcoming_race_df)
        
        # Format predictions for JSON storage
        predictions_list = []
        for _, row in predictions.iterrows():
            predictions_list.append({
                'rank': int(row.name + 1),
                'horse_name': row['horse_name'],
                'barrier': horse_race_details.get(row['horse_name'], {}).get('barrier'),
                'weight': horse_race_details.get(row['horse_name'], {}).get('weight'),
                'jockey': horse_race_details.get(row['horse_name'], {}).get('jockey'),
                'trainer': horse_race_details.get(row['horse_name'], {}).get('trainer'),
                'win_probability': float(row['win_probability']),
                'place_probability': float(row['place_probability']),
                'predicted_finish': float(row['predicted_finish']),
                'confidence': float(row['confidence'])
            })
        
        return {
            'race_info': race_info,
            'predictions': predictions_list,
            'model_metrics': {
                'training_records': len(training_records),
                'field_size': len(target_horses),
                'win_auc': float(metrics.get('win_auc', 0.0)),
                'place_auc': float(metrics.get('place_auc', 0.0)),
                'finish_rmse': float(metrics.get('finish_rmse', 0.0)),
                'features_used': int(metrics.get('features_used', 0))
            },
            'created_at': datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Prediction generation failed: {e}")
        raise

@app.post("/predict", response_model=PredictionResponse)
async def create_prediction(request: PredictionRequest, background_tasks: BackgroundTasks):
    """
    Generate a new race prediction.
    
    Creates neural network predictions for the specified race and stores the results.
    """
    try:
        # Generate unique prediction ID
        prediction_id = str(uuid.uuid4())
        
        # Log the request
        logger.info(f"Generating prediction {prediction_id} for {request.date} {request.venue} Race {request.race_number}")
        
        # Generate prediction using core logic
        prediction_data = generate_prediction_core(
            request.date, 
            request.venue, 
            request.race_number, 
            request.api_key
        )
        
        # Add metadata
        prediction_data['prediction_id'] = prediction_id
        prediction_data['request'] = request.dict(exclude={'api_key'})  # Don't store API key
        
        # Store prediction in Cosmos DB or fallback to memory
        if cosmos_client:
            cosmos_client.store_prediction(prediction_id, prediction_data)
        else:
            prediction_store[prediction_id] = prediction_data
        
        logger.info(f"Prediction {prediction_id} completed successfully")
        
        return PredictionResponse(
            prediction_id=prediction_id,
            status="completed",
            message="Prediction generated successfully",
            race_info=prediction_data['race_info'],
            predictions=prediction_data['predictions'],
            model_metrics=prediction_data['model_metrics'],
            created_at=prediction_data['created_at']
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Internal error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error occurred")

@app.get("/predictions/{prediction_id}", response_model=PredictionResponse)
async def get_prediction(prediction_id: str):
    """
    Retrieve a stored prediction by ID.
    """
    # Try Cosmos DB first, then fallback to memory
    prediction_data = None
    
    if cosmos_client:
        prediction_data = cosmos_client.get_prediction(prediction_id)
    
    if not prediction_data and prediction_id in prediction_store:
        prediction_data = prediction_store[prediction_id]
    
    if not prediction_data:
        raise HTTPException(status_code=404, detail="Prediction not found")
    
    return PredictionResponse(
        prediction_id=prediction_id,
        status="completed",
        message="Prediction retrieved successfully", 
        race_info=prediction_data['race_info'],
        predictions=prediction_data['predictions'],
        model_metrics=prediction_data['model_metrics'],
        created_at=prediction_data['created_at']
    )

@app.get("/predictions/search", response_model=PredictionSearchResponse)
async def search_predictions(
    date: Optional[str] = None,
    venue: Optional[str] = None,
    race_number: Optional[int] = None,
    limit: int = Field(default=20, ge=1, le=100)
):
    """
    Search for predictions by criteria.
    """
    matching_predictions = []
    
    # Try Cosmos DB first
    if cosmos_client:
        cosmos_results = cosmos_client.search_predictions(date, venue, race_number, limit)
        for result in cosmos_results:
            matching_predictions.append({
                'prediction_id': result.get('prediction_id'),
                'race_info': result.get('race_info', {}),
                'created_at': result.get('created_at'),
                'field_size': result.get('model_metrics', {}).get('field_size', 0)
            })
    else:
        # Fallback to in-memory search
        for pred_id, pred_data in prediction_store.items():
            race_info = pred_data.get('race_info', {})
            
            # Apply filters
            if date and race_info.get('date') != date:
                continue
            if venue and venue.lower() not in race_info.get('venue', '').lower():
                continue
            if race_number and race_info.get('race_number') != race_number:
                continue
                
            matching_predictions.append({
                'prediction_id': pred_id,
                'race_info': race_info,
                'created_at': pred_data.get('created_at'),
                'field_size': pred_data.get('model_metrics', {}).get('field_size', 0)
            })
        
        # Sort by creation time (newest first)
        matching_predictions.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        # Apply limit
        matching_predictions = matching_predictions[:limit]
    
    return PredictionSearchResponse(
        predictions=matching_predictions,
        total_count=len(matching_predictions)
    )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}

if __name__ == "__main__":
    # Get configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    logger.info(f"Starting Neural Racing Predictor API on {host}:{port}")
    
    uvicorn.run(
        "api_main:app",
        host=host,
        port=port,
        reload=True,  # Disable in production
        log_level="info"
    )