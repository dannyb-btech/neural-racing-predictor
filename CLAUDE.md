# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A neural network-based horse racing prediction system that uses racing.com API data to generate win/place probabilities and expected finishing positions. The system extracts historical race data, trains a multi-task neural network, and provides calibrated probability predictions.

## Development Commands

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run predictions:**
```bash
python neural_racing_predictor.py 2025-07-16 "Sportsbet Sandown Hillside" 8 --api-key YOUR_KEY
```

**Environment setup:**
Create `.env` file with:
```
RACING_API_KEY=your_api_key_here
```

## Architecture

### Core Components

**neural_racing_predictor.py** - Main orchestration script that:
- Coordinates data extraction, model training, and prediction
- Handles command-line arguments and environment setup
- Prints formatted race summaries and predictions

**racing_com_api_client.py** - GraphQL API client for racing.com:
- Implements GraphQL queries for meetings, races, entries, and horse form
- Handles rate limiting and API authentication
- Provides speed map data from separate FAIS API endpoint

**feature_creation.py** - Bayesian training data extraction:
- `BayesianTrainingDataExtractor` class extracts historical race records
- `HistoricalRaceRecord` dataclass defines training data structure
- Handles timing differences, career statistics, and performance metrics

**neural_racing_model.py** - Neural network implementation:
- `HorseRacingNet` class implements multi-task PyTorch model
- `NeuralRacingModel` wrapper handles training, prediction, and calibration
- Predicts win probability, place probability, and expected finish position

### Data Flow

1. **Data Extraction**: API client fetches historical race data and horse form
2. **Feature Engineering**: Training data extractor creates standardized features from raw race data
3. **Model Training**: Neural network trains on historical performance with minimum 50 records
4. **Prediction**: Model generates calibrated probabilities for upcoming races

### Key Technical Details

- Multi-task learning with shared feature extraction layers
- Handles horses with no mutual racing history through standardized performance metrics
- Uses PyTorch for neural network implementation with StandardScaler for feature normalization
- Requires minimum 50 training records for model training
- Provides confidence intervals alongside predictions