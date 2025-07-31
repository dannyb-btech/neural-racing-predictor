#!/usr/bin/env python3
"""
Test script to verify Cosmos DB connection with existing database and container.
"""

import logging
from datetime import datetime, timezone
from cosmos_db_client import CosmosDBClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_cosmos_connection():
    """Test the Cosmos DB connection and basic operations."""
    
    print("🔍 Testing Cosmos DB Connection")
    print("=" * 50)
    
    try:
        # Initialize Cosmos DB client (uses .env configuration)
        cosmos_client = CosmosDBClient()
        print("✅ Cosmos DB client initialized successfully")
        
        # Test data
        test_prediction_id = f"test-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        test_prediction_data = {
            'prediction_id': test_prediction_id,
            'race_info': {
                'date': '2025-07-25',
                'venue': 'Test Venue',
                'race_number': 1,
                'distance': 1200,
                'track_condition': 'Good',
                'class': 'Test Class'
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
                'training_records': 1000,
                'field_size': 8,
                'win_auc': 0.75,
                'place_auc': 0.82
            },
            'created_at': datetime.now(timezone.utc).isoformat()
        }
        
        # Test 1: Store prediction
        print(f"\n📝 Testing prediction storage...")
        success = cosmos_client.store_prediction(test_prediction_id, test_prediction_data)
        if success:
            print("✅ Prediction stored successfully")
        else:
            print("❌ Failed to store prediction")
            return False
        
        # Test 2: Retrieve prediction
        print(f"\n📖 Testing prediction retrieval...")
        retrieved_data = cosmos_client.get_prediction(test_prediction_id)
        if retrieved_data:
            print("✅ Prediction retrieved successfully")
            print(f"   Race: {retrieved_data['race_info']['venue']} Race {retrieved_data['race_info']['race_number']}")
            print(f"   Predictions: {len(retrieved_data['predictions'])} horses")
        else:
            print("❌ Failed to retrieve prediction")
            return False
        
        # Test 3: Search predictions
        print(f"\n🔍 Testing prediction search...")
        search_results = cosmos_client.search_predictions(date='2025-07-25', limit=5)
        print(f"✅ Found {len(search_results)} predictions for 2025-07-25")
        
        # Test 4: Get database stats
        print(f"\n📊 Testing database statistics...")
        stats = cosmos_client.get_database_stats()
        if stats:
            print("✅ Database statistics retrieved:")
            print(f"   Total predictions: {stats.get('total_predictions', 'N/A')}")
            print(f"   Database: {stats.get('database_name', 'N/A')}")
            print(f"   Container: {stats.get('container_name', 'N/A')}")
        
        # Test 5: Cleanup test data
        print(f"\n🧹 Cleaning up test data...")
        # Note: For cleanup, we need the partition key value
        # This depends on your container's partition key configuration
        print(f"   Test prediction ID: {test_prediction_id}")
        print("   (Manual cleanup may be required depending on partition key configuration)")
        
        print("\n🎉 All Cosmos DB tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Cosmos DB test failed: {e}")
        logger.error(f"Test failed with error: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    test_cosmos_connection()