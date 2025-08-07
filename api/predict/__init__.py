import azure.functions as func
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
import pandas as pd

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Predict endpoint called.')
    
    try:
        # Parse request body
        req_body = req.get_json()
        if not req_body:
            return func.HttpResponse(
                json.dumps({"detail": "Request body is required"}),
                status_code=400,
                headers={"Content-Type": "application/json"}
            )
        
        # Extract request parameters
        date = req_body.get('date')
        venue = req_body.get('venue')
        race_number = req_body.get('race_number')
        use_neural_network = req_body.get('use_neural_network', False)
        
        if not all([date, venue, race_number]):
            return func.HttpResponse(
                json.dumps({"detail": "date, venue, and race_number are required"}),
                status_code=400,
                headers={"Content-Type": "application/json"}
            )
        
        # Get API key from environment
        api_key = os.getenv('RACING_API_KEY')
        if not api_key:
            return func.HttpResponse(
                json.dumps({"detail": "Racing API key not configured"}),
                status_code=500,
                headers={"Content-Type": "application/json"}
            )
        
        # Initialize Cosmos DB client
        from cosmos_db_client import CosmosDBClient
        cosmos_client = CosmosDBClient()
        
        # Generate prediction ID
        prediction_id = str(uuid.uuid4())
        
        if use_neural_network:
            # Generate real neural network prediction
            logging.info(f"Generating neural network prediction for {venue} Race {race_number}")
            prediction_data = generate_neural_prediction(date, venue, race_number, api_key)
            prediction_data['prediction_id'] = prediction_id
        else:
            # Generate test prediction
            prediction_data = {
                'prediction_id': prediction_id,
                'race_info': {
                    'date': date,
                    'venue': venue,
                    'race_number': race_number,
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
        
        # Store in Cosmos DB
        success = cosmos_client.store_prediction(prediction_id, prediction_data)
        if not success:
            logging.error("Failed to store prediction in Cosmos DB")
            return func.HttpResponse(
                json.dumps({"detail": "Failed to store prediction"}),
                status_code=500,
                headers={"Content-Type": "application/json"}
            )
        
        return func.HttpResponse(
            json.dumps({
                'prediction_id': prediction_id,
                'status': 'completed',
                'message': 'Prediction created successfully',
                'data': prediction_data
            }),
            status_code=200,
            headers={"Content-Type": "application/json"}
        )
        
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return func.HttpResponse(
            json.dumps({"detail": f"Prediction failed: {str(e)}"}),
            status_code=500,
            headers={"Content-Type": "application/json"}
        )

def generate_neural_prediction(date: str, venue: str, race_number: int, api_key: str) -> Dict[str, Any]:
    """Generate real neural network predictions."""
    try:
        # Import required modules
        from racing_com_api_client import RacingComAPIClient
        from feature_creation import BayesianTrainingDataExtractor, RaceTarget
        from neural_racing_model import NeuralRacingModel
        from neural_racing_predictor import create_upcoming_race_data
        
        # Step 1: Extract Training Data
        logging.info(f"Extracting training data for {venue} Race {race_number} on {date}")
        
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
        
        logging.info(f"Extracted {len(training_records)} training records for {len(target_horses)} horses")
        
        # Step 2: Train Neural Network
        logging.info("Training neural network...")
        
        model = NeuralRacingModel(
            hidden_dims=[128, 64, 32],
            learning_rate=0.001,
            device='cpu'
        )
        
        metrics = model.train(
            df, 
            epochs=50,
            batch_size=32,
            validation_split=0.2
        )
        
        logging.info("Neural network training completed")
        
        # Step 3: Generate Predictions
        logging.info("Generating race predictions...")
        
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
            
            logging.info("Extracted race details from API")
            
        except Exception as e:
            logging.warning(f"Could not fetch race details from API: {e}")
        
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
        
        logging.info(f"Generated predictions for {len(predictions_list)} horses")
        
        # Determine prediction strength
        val_win_auc = float(metrics.get('val_win_auc', metrics.get('win_auc', 0.0)))
        if val_win_auc > 0.65:
            strength_assessment = "Good"
        elif val_win_auc > 0.55:
            strength_assessment = "Moderate"
        else:
            strength_assessment = "Poor"
        
        return {
            'race_info': race_info,
            'predictions': predictions_list,
            'model_metrics': {
                'training_records': len(training_records),
                'field_size': len(target_horses),
                'win_auc': float(metrics.get('win_auc', 0.0)),
                'place_auc': float(metrics.get('place_auc', 0.0)),
                'val_win_auc': val_win_auc,
                'val_place_auc': float(metrics.get('val_place_auc', metrics.get('place_auc', 0.0))),
                'win_loss': float(metrics.get('win_loss', 0.0)),
                'place_loss': float(metrics.get('place_loss', 0.0)),
                'prediction_strength': strength_assessment
            },
            'created_at': datetime.now(timezone.utc).isoformat(),
            'note': 'Neural network prediction using historical racing data'
        }
        
    except Exception as e:
        raise Exception(f"Neural prediction failed: {str(e)}")