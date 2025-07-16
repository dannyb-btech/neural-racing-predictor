# Neural Racing Predictor

A neural network-based horse racing prediction system that uses racing.com API data to generate win/place probabilities and expected finishing positions.

## Files

- **neural_racing_predictor.py** - Main prediction script
- **racing_com_api_client.py** - Racing.com GraphQL API client
- **feature_creation.py** - Feature extraction and training data preparation
- **neural_racing_model.py** - Neural network model implementation
- **.env** - Environment variables (API key)
- **requirements.txt** - Python dependencies

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up API key in .env file:
```
RACING_API_KEY=your_api_key_here
```

## Usage

```bash
python neural_racing_predictor.py 2025-07-16 "Sportsbet Sandown Hillside" 8 --api-key YOUR_KEY
```

## Features

- Extracts historical race data from racing.com API
- Uses neural network for multi-task learning (win/place/finish position)
- Handles horses that have never raced against each other
- Provides calibrated probabilities with confidence intervals
- Shows actual race entry numbers from API data

## Model

- Multi-task neural network with shared feature extraction
- Minimum 50 training records required
- Uses standardized timing differences and career statistics
- Outputs win probability, place probability, and expected finish position