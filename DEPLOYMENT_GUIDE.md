# Neural Racing Predictor API & Web App Deployment Guide

This guide covers deploying the Neural Racing Predictor as a complete web application with API backend and static frontend.

## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│                 │    │                  │    │                 │
│ Static Web App  │───▶│   FastAPI        │───▶│   Cosmos DB     │
│ (Frontend)      │    │   (Backend)      │    │   (Storage)     │
│                 │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Prerequisites

1. **Azure Account** with active subscription
2. **Azure CLI** installed and authenticated
3. **Python 3.8+** for local development
4. **Racing.com API Key** for race data access
5. **Git** for version control

## Part 1: Azure Cosmos DB Configuration

### 1.1 Using Existing Cosmos DB

This system uses your existing Cosmos DB instance configured in the `.env` file:

- **Database**: `HorseRacingDB` (from COSMOS_DATABASE_ID)
- **Container**: `RaceAnalysis` (from COSMOS_CONTAINER_ID)
- **Endpoint**: From COSMOS_ENDPOINT environment variable
- **Key**: From COSMOS_KEY environment variable

### 1.2 Environment Variables Required

Ensure your `.env` file contains:

```bash
COSMOS_ENDPOINT=https://your-cosmos-account.documents.azure.com:443/
COSMOS_KEY=your-cosmos-primary-key
COSMOS_DATABASE_ID=HorseRacingDB
COSMOS_CONTAINER_ID=RaceAnalysis
RACING_API_KEY=your-racing-api-key
```

### 1.3 Document Structure

Predictions are stored with the following structure:

```json
{
  "id": "unique-prediction-id",
  "type": "race_prediction",
  "race_date": "2025-07-25",
  "venue": "Flemington",
  "race_number": 7,
  "race_info": { ... },
  "predictions": [ ... ],
  "model_metrics": { ... },
  "created_at": "2025-07-25T10:30:00Z"
}
```

## Part 2: FastAPI Backend Deployment

### 2.1 Deploy to Azure Container Apps

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY api_requirements.txt .
RUN pip install --no-cache-dir -r api_requirements.txt

# Copy application code
COPY *.py ./
COPY neural_predictions/ ./neural_predictions/

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "api_main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Deploy with Azure Container Apps:

```bash
# Create Container Apps environment
az containerapp env create \
  --resource-group racing-predictor-rg \
  --name racing-predictor-env \
  --location australiaeast

# Deploy the API
az containerapp create \
  --resource-group racing-predictor-rg \
  --name racing-predictor-api \
  --environment racing-predictor-env \
  --image YOUR_REGISTRY/racing-predictor-api:latest \
  --target-port 8000 \
  --ingress external \
  --min-replicas 1 \
  --max-replicas 3 \
  --cpu 1.0 \
  --memory 2.0Gi \
  --env-vars COSMOS_DB_CONNECTION_STRING="YOUR_CONNECTION_STRING"
```

### 2.2 Alternative: Azure App Service

```bash
# Create App Service plan
az appservice plan create \
  --resource-group racing-predictor-rg \
  --name racing-predictor-plan \
  --sku B1 \
  --is-linux

# Create web app
az webapp create \
  --resource-group racing-predictor-rg \
  --plan racing-predictor-plan \
  --name racing-predictor-api \
  --runtime "PYTHON|3.11"

# Configure environment variables
az webapp config appsettings set \
  --resource-group racing-predictor-rg \
  --name racing-predictor-api \
  --settings COSMOS_DB_CONNECTION_STRING="YOUR_CONNECTION_STRING"

# Deploy code (using zip deployment)
zip -r app.zip *.py api_requirements.txt neural_predictions/
az webapp deployment source config-zip \
  --resource-group racing-predictor-rg \
  --name racing-predictor-api \
  --src app.zip
```

## Part 3: Static Web App Frontend

### 3.1 Deploy to Azure Static Web Apps

```bash
# Create Static Web App
az staticwebapp create \
  --resource-group racing-predictor-rg \
  --name racing-predictor-web \
  --source https://github.com/YOUR_USERNAME/neural-racing-predictor \
  --branch main \
  --app-location "/web-app" \
  --api-location "/api" \
  --output-location "/"
```

### 3.2 Configure API Endpoint

Update the `API_BASE_URL` in `web-app/index.html`:

```javascript
const API_BASE_URL = 'https://racing-predictor-api.azurecontainerapps.io'; // Your API URL
```

### 3.3 Custom Domain (Optional)

```bash
# Add custom domain
az staticwebapp hostname set \
  --resource-group racing-predictor-rg \
  --name racing-predictor-web \
  --hostname yourdomain.com
```

## Part 4: Local Development Setup

### 4.1 Environment Setup

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/neural-racing-predictor.git
cd neural-racing-predictor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r api_requirements.txt

# Create .env file
echo "COSMOS_DB_CONNECTION_STRING=your_connection_string_here" > .env
echo "RACING_API_KEY=your_racing_api_key_here" >> .env
```

### 4.2 Run Local Development

```bash
# Start FastAPI server
uvicorn api_main:app --reload --host 0.0.0.0 --port 8000

# In another terminal, serve the web app
cd web-app
python -m http.server 3000
```

Access the application at:
- API: http://localhost:8000
- Web App: http://localhost:3000
- API Docs: http://localhost:8000/docs

## Part 5: API Usage Examples

### 5.1 Generate Prediction

```bash
curl -X POST "https://your-api-url.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "date": "2025-07-25",
    "venue": "Flemington",
    "race_number": 7,
    "api_key": "your_racing_api_key"
  }'
```

### 5.2 Retrieve Prediction

```bash
curl -X GET "https://your-api-url.com/predictions/{prediction_id}"
```

### 5.3 Search Predictions

```bash
curl -X GET "https://your-api-url.com/predictions/search?date=2025-07-25&venue=Flemington"
```

## Part 6: Monitoring & Maintenance

### 6.1 Application Insights

```bash
# Create Application Insights
az monitor app-insights component create \
  --resource-group racing-predictor-rg \
  --app racing-predictor-insights \
  --location australiaeast \
  --kind web

# Get instrumentation key
az monitor app-insights component show \
  --resource-group racing-predictor-rg \
  --app racing-predictor-insights \
  --query instrumentationKey
```

### 6.2 Logging Configuration

Add to your API environment variables:
```bash
APPLICATIONINSIGHTS_CONNECTION_STRING="your_connection_string"
```

### 6.3 Health Monitoring

The API includes a health endpoint at `/health` for monitoring:

```bash
curl https://your-api-url.com/health
```

## Part 7: Security Considerations

### 7.1 API Key Management

- Store API keys in Azure Key Vault
- Use managed identities for Azure resource access
- Implement rate limiting for the API

### 7.2 CORS Configuration

Update CORS settings in `api_main.py` for production:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Specific domain
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

### 7.3 Input Validation

The API includes comprehensive input validation using Pydantic models.

## Part 8: Cost Optimization

### 8.1 Cosmos DB
- Use Serverless billing mode for variable workloads
- Set up auto-scale based on usage patterns
- Implement data retention policies

### 8.2 Container Apps / App Service
- Use consumption-based pricing
- Set up auto-scaling rules
- Monitor resource usage

### 8.3 Static Web Apps
- Static Web Apps include generous free tier
- CDN caching reduces data transfer costs

## Troubleshooting

### Common Issues

1. **CORS Errors**: Check CORS configuration in API
2. **Connection Timeouts**: Increase timeout settings for neural network processing
3. **Memory Issues**: Ensure sufficient memory allocation for model training
4. **API Rate Limits**: Implement caching for repeated requests

### Logs and Debugging

```bash
# View Container Apps logs
az containerapp logs show \
  --resource-group racing-predictor-rg \
  --name racing-predictor-api

# View App Service logs
az webapp log tail \
  --resource-group racing-predictor-rg \
  --name racing-predictor-api
```

## Production Checklist

- [ ] Environment variables configured
- [ ] HTTPS enabled
- [ ] Custom domain configured (optional)
- [ ] Monitoring and alerts set up
- [ ] Backup and disaster recovery planned
- [ ] Rate limiting implemented
- [ ] Security headers configured
- [ ] Performance testing completed
- [ ] Documentation updated

## Support

For issues and questions:
1. Check the API documentation at `/docs` endpoint
2. Review application logs
3. Verify Azure resource configuration
4. Test with sample API calls

## Cost Estimates

**Monthly estimates for moderate usage:**
- Cosmos DB Serverless: $10-50
- Container Apps: $15-30
- Static Web Apps: Free (within limits)
- **Total**: ~$25-80/month

Costs scale with usage and can be optimized based on actual demand patterns.