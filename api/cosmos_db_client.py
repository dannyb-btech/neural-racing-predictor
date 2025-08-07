#!/usr/bin/env python3
"""
Azure Cosmos DB client for storing and retrieving racing predictions.
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

from azure.cosmos import CosmosClient, PartitionKey, exceptions
from azure.cosmos.database import DatabaseProxy
from azure.cosmos.container import ContainerProxy

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class CosmosDBClient:
    """
    Azure Cosmos DB client for racing predictions storage.
    """
    
    def __init__(self, endpoint: str = None, key: str = None, database_name: str = None, container_name: str = None):
        """
        Initialize Cosmos DB client using endpoint and key.
        
        Args:
            endpoint: Cosmos DB endpoint URL (from environment if not provided)
            key: Cosmos DB primary key (from environment if not provided)
            database_name: Name of the database to use (from environment if not provided)
            container_name: Name of the container to use (from environment if not provided)
        """
        # Get configuration from environment if not provided
        self.endpoint = endpoint or os.getenv('COSMOS_ENDPOINT')
        self.key = key or os.getenv('COSMOS_KEY')
        self.database_name = database_name or os.getenv('COSMOS_DATABASE_ID', 'HorseRacingDB')
        self.container_name = container_name or 'RacePredictions'  # New container for predictions
        
        if not self.endpoint:
            raise ValueError("Cosmos DB endpoint not provided and COSMOS_ENDPOINT environment variable not set")
        if not self.key:
            raise ValueError("Cosmos DB key not provided and COSMOS_KEY environment variable not set")
        
        # Initialize Cosmos client
        self.client = CosmosClient(self.endpoint, self.key)
        
        # Initialize database and container
        self._setup_database_and_container()
    
    def _setup_database_and_container(self):
        """Set up database and container, creating them if they don't exist."""
        try:
            # Get or create database
            self.database = self.client.create_database_if_not_exists(id=self.database_name)
            logger.info(f"Database '{self.database_name}' ready")
            
            # Create container for predictions with appropriate partition key
            # Using /race_date as partition key for time-based partitioning
            # No throughput setting needed for serverless accounts
            self.container = self.database.create_container_if_not_exists(
                id=self.container_name,
                partition_key=PartitionKey(path="/race_date")
            )
            logger.info(f"Container '{self.container_name}' ready")
            
            # Read container properties to confirm setup
            container_props = self.container.read()
            logger.info(f"Container partition key: {container_props.get('partitionKey', {}).get('paths', ['N/A'])}")
            
        except exceptions.CosmosHttpResponseError as e:
            logger.error(f"Failed to setup Cosmos DB: {e}")
            raise
    
    def store_prediction(self, prediction_id: str, prediction_data: Dict[str, Any]) -> bool:
        """
        Store a prediction in Cosmos DB.
        
        Args:
            prediction_id: Unique identifier for the prediction
            prediction_data: Complete prediction data including race_info, predictions, etc.
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Prepare document for storage
            race_info = prediction_data.get('race_info', {})
            race_date = race_info.get('date', datetime.now().strftime('%Y-%m-%d'))
            
            document = {
                'id': prediction_id,
                'prediction_id': prediction_id,
                'type': 'race_prediction',
                'race_date': race_date,  # Partition key
                'created_at': prediction_data.get('created_at', datetime.now(timezone.utc).isoformat()),
                'race_info': race_info,
                'predictions': prediction_data.get('predictions', []),
                'model_metrics': prediction_data.get('model_metrics', {}),
                'request': prediction_data.get('request', {}),
                # Additional queryable fields
                'venue': race_info.get('venue'),
                'race_number': race_info.get('race_number'),
                'field_size': prediction_data.get('model_metrics', {}).get('field_size', 0),
                '_ts': int(datetime.now(timezone.utc).timestamp())
            }
            
            # Store in Cosmos DB
            self.container.create_item(body=document)
            logger.info(f"Prediction {prediction_id} stored successfully in Cosmos DB")
            return True
            
        except exceptions.CosmosHttpResponseError as e:
            logger.error(f"Failed to store prediction {prediction_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error storing prediction {prediction_id}: {e}")
            return False
    
    def get_prediction(self, prediction_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a prediction from Cosmos DB.
        
        Args:
            prediction_id: Unique identifier for the prediction
            
        Returns:
            Dict containing prediction data or None if not found
        """
        try:
            # Query for the prediction (enable cross-partition query)
            query = "SELECT * FROM c WHERE c.prediction_id = @prediction_id"
            parameters = [{"name": "@prediction_id", "value": prediction_id}]
            
            items = list(self.container.query_items(
                query=query, 
                parameters=parameters,
                enable_cross_partition_query=True
            ))
            
            if not items:
                logger.warning(f"Prediction {prediction_id} not found")
                return None
            
            # Return the first (and should be only) match
            prediction_data = items[0]
            logger.info(f"Prediction {prediction_id} retrieved successfully")
            return prediction_data
            
        except exceptions.CosmosHttpResponseError as e:
            logger.error(f"Failed to retrieve prediction {prediction_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error retrieving prediction {prediction_id}: {e}")
            return None
    
    def search_predictions(self, date: str = None, venue: str = None, race_number: int = None, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Search for predictions by criteria.
        
        Args:
            date: Race date filter (YYYY-MM-DD)
            venue: Venue name filter (case-insensitive partial match)
            race_number: Race number filter
            limit: Maximum number of results to return
            
        Returns:
            List of matching prediction documents
        """
        try:
            # Build dynamic query
            query_parts = ["SELECT * FROM c WHERE c.type = 'race_prediction'"]
            parameters = []
            
            if date:
                query_parts.append("AND c.race_info.date = @date")
                parameters.append({"name": "@date", "value": date})
            
            if venue:
                query_parts.append("AND CONTAINS(LOWER(c.race_info.venue), @venue)")
                parameters.append({"name": "@venue", "value": venue.lower()})
            
            if race_number:
                query_parts.append("AND c.race_info.race_number = @race_number")
                parameters.append({"name": "@race_number", "value": race_number})
            
            # Add ordering and limit
            query_parts.append("ORDER BY c._ts DESC")
            
            query = " ".join(query_parts)
            
            # Execute query
            items = list(self.container.query_items(
                query=query, 
                parameters=parameters,
                max_item_count=limit,
                enable_cross_partition_query=True
            ))
            
            logger.info(f"Found {len(items)} predictions matching search criteria")
            return items
            
        except exceptions.CosmosHttpResponseError as e:
            logger.error(f"Failed to search predictions: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error searching predictions: {e}")
            return []
    
    def get_recent_predictions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the most recent predictions.
        
        Args:
            limit: Maximum number of predictions to return
            
        Returns:
            List of recent prediction documents
        """
        try:
            query = "SELECT * FROM c WHERE c.type = 'race_prediction' ORDER BY c._ts DESC"
            
            items = list(self.container.query_items(
                query=query,
                max_item_count=limit,
                enable_cross_partition_query=True
            ))
            
            logger.info(f"Retrieved {len(items)} recent predictions")
            return items
            
        except exceptions.CosmosHttpResponseError as e:
            logger.error(f"Failed to get recent predictions: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error getting recent predictions: {e}")
            return []
    
    def delete_prediction(self, prediction_id: str, partition_key: str) -> bool:
        """
        Delete a prediction from Cosmos DB.
        
        Args:
            prediction_id: Unique identifier for the prediction
            partition_key: Partition key value (race date)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.container.delete_item(item=prediction_id, partition_key=partition_key)
            logger.info(f"Prediction {prediction_id} deleted successfully")
            return True
            
        except exceptions.CosmosResourceNotFoundError:
            logger.warning(f"Prediction {prediction_id} not found for deletion")
            return False
        except exceptions.CosmosHttpResponseError as e:
            logger.error(f"Failed to delete prediction {prediction_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error deleting prediction {prediction_id}: {e}")
            return False
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dictionary containing database statistics
        """
        try:
            # Count total predictions
            count_query = "SELECT VALUE COUNT(1) FROM c WHERE c.type = 'race_prediction'"
            count_result = list(self.container.query_items(
                query=count_query,
                enable_cross_partition_query=True
            ))
            total_predictions = count_result[0] if count_result else 0
            
            # Get date range
            date_query = """
                SELECT 
                    MIN(c.race_info.date) as min_date,
                    MAX(c.race_info.date) as max_date
                FROM c WHERE c.type = 'race_prediction'
            """
            date_result = list(self.container.query_items(
                query=date_query,
                enable_cross_partition_query=True
            ))
            date_range = date_result[0] if date_result else {}
            
            return {
                'total_predictions': total_predictions,
                'date_range': date_range,
                'database_name': self.database_name,
                'container_name': self.container_name
            }
            
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {}

# Example usage and testing
if __name__ == "__main__":
    # This would be used for testing the Cosmos DB client
    import json
    
    # Example prediction data structure
    sample_prediction = {
        'prediction_id': 'test-123',
        'race_info': {
            'date': '2025-07-25',
            'venue': 'Flemington',
            'race_number': 7,
            'distance': 1600,
            'track_condition': 'Good',
            'class': 'Group 1'
        },
        'predictions': [
            {
                'rank': 1,
                'horse_name': 'Test Horse',
                'win_probability': 0.25,
                'place_probability': 0.65,
                'predicted_finish': 2.1,
                'confidence': 0.85
            }
        ],
        'model_metrics': {
            'training_records': 1500,
            'field_size': 8,
            'win_auc': 0.78,
            'place_auc': 0.85
        },
        'created_at': datetime.now(timezone.utc).isoformat()
    }
    
    print("Cosmos DB client ready for integration")
    print("Sample prediction structure:", json.dumps(sample_prediction, indent=2))