import azure.functions as func
import json
import logging

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Get prediction by ID endpoint called.')
    
    try:
        # Get prediction ID from route parameters
        prediction_id = req.route_params.get('prediction_id')
        
        if not prediction_id:
            return func.HttpResponse(
                json.dumps({"detail": "prediction_id is required"}),
                status_code=400,
                headers={"Content-Type": "application/json"}
            )
        
        logging.info(f"Retrieving prediction: {prediction_id}")
        
        # Initialize Cosmos DB client
        from cosmos_db_client import CosmosDBClient
        cosmos_client = CosmosDBClient()
        
        # Get prediction
        prediction = cosmos_client.get_prediction(prediction_id)
        
        if not prediction:
            return func.HttpResponse(
                json.dumps({"detail": "Prediction not found"}),
                status_code=404,
                headers={"Content-Type": "application/json"}
            )
        
        logging.info(f"Found prediction: {prediction_id}")
        
        return func.HttpResponse(
            json.dumps(prediction),
            status_code=200,
            headers={"Content-Type": "application/json"}
        )
        
    except Exception as e:
        logging.error(f"Get prediction error: {str(e)}")
        return func.HttpResponse(
            json.dumps({
                "detail": f"Failed to retrieve prediction: {str(e)}"
            }),
            status_code=500,
            headers={"Content-Type": "application/json"}
        )