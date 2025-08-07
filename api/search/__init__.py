import azure.functions as func
import json
import logging
import os

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Search predictions endpoint called.')
    
    try:
        # Get query parameters
        date = req.params.get('date')
        venue = req.params.get('venue')
        
        logging.info(f"Search params - date: {date}, venue: {venue}")
        
        # Initialize Cosmos DB client
        from cosmos_db_client import CosmosDBClient
        cosmos_client = CosmosDBClient()
        
        # Search predictions
        predictions = cosmos_client.search_predictions(date=date, venue=venue, limit=100)
        
        logging.info(f"Found {len(predictions)} predictions")
        
        return func.HttpResponse(
            json.dumps({
                'predictions': predictions,
                'total': len(predictions)
            }),
            status_code=200,
            headers={"Content-Type": "application/json"}
        )
        
    except Exception as e:
        logging.error(f"Search error: {str(e)}")
        return func.HttpResponse(
            json.dumps({
                "detail": f"Search failed: {str(e)}"
            }),
            status_code=500,
            headers={"Content-Type": "application/json"}
        )