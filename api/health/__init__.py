import azure.functions as func
import json
import logging
from datetime import datetime, timezone

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Health check endpoint called.')
    
    try:
        # Test Cosmos DB connection
        cosmos_status = "disconnected"
        try:
            from cosmos_db_client import CosmosDBClient
            cosmos_client = CosmosDBClient()
            stats = cosmos_client.get_database_stats()
            cosmos_status = "connected"
        except Exception as e:
            logging.warning(f"Cosmos DB connection test failed: {e}")
            cosmos_status = "error"
        
        return func.HttpResponse(
            json.dumps({
                "status": "ok",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "cosmos_db": cosmos_status,
                "service": "Neural Racing Predictor API"
            }),
            status_code=200,
            headers={"Content-Type": "application/json"}
        )
    except Exception as e:
        logging.error(f"Health check failed: {e}")
        return func.HttpResponse(
            json.dumps({
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }),
            status_code=500,
            headers={"Content-Type": "application/json"}
        )