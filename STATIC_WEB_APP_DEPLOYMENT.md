# Step-by-Step Static Web App Deployment Guide

This guide provides detailed steps to deploy your Neural Racing Predictor web application to Azure Static Web Apps.

## Prerequisites

- Azure subscription
- GitHub account
- Your project pushed to a GitHub repository
- FastAPI backend already deployed (or running locally for testing)

## Step 1: Prepare Your Repository

### 1.1 Ensure Correct Folder Structure

Your repository should look like this:
```
neural-racing-predictor/
├── web-app/
│   └── index.html          # Your static web app
├── simple_api.py           # FastAPI backend
├── requirements.txt        # Python dependencies
└── other files...
```

### 1.2 Commit and Push to GitHub

```bash
git add .
git commit -m "Prepare for Azure Static Web App deployment"
git push origin main
```

## Step 2: Create Azure Static Web App via Portal

### 2.1 Sign in to Azure Portal

1. Go to https://portal.azure.com
2. Sign in with your Azure account

### 2.2 Create Static Web App Resource

1. Click **"Create a resource"** (+ symbol)
2. Search for **"Static Web App"**
3. Click **"Static Web App"** from Microsoft
4. Click **"Create"**

### 2.3 Configure Basic Settings

**Subscription & Resource Group:**
- **Subscription**: Select your Azure subscription
- **Resource Group**: Create new or select existing (e.g., `neural-racing-rg`)

**Static Web App Details:**
- **Name**: `neural-racing-predictor` (must be globally unique)
- **Plan type**: 
  - **Free** for development/testing
  - **Standard** for production (includes custom domains, SLA)
- **Azure Functions and staging environments**: Leave as default
- **Region**: Choose closest to your users (e.g., `East US 2`)

### 2.4 Configure Deployment Source

**GitHub Integration:**
1. **Source**: Select **"GitHub"**
2. Click **"Sign in with GitHub"** (authorize Azure to access your repos)
3. **Organization**: Select your GitHub username/organization
4. **Repository**: Select `neural-racing-predictor` (or your repo name)
5. **Branch**: Select `main` (or your deployment branch)

**Build Details:**
- **Build Presets**: Select **"Custom"**
- **App location**: `/web-app`
- **Api location**: `/api` (this will contain our Azure Functions)
- **Output location**: Leave **empty**

### 2.5 Review and Create

1. Click **"Review + create"**
2. Review your settings
3. Click **"Create"**

## Step 3: Monitor Deployment

### 3.1 Check Deployment Status

1. After creation, go to your Static Web App resource
2. Click **"GitHub Action runs"** to see deployment progress
3. Wait for the initial deployment to complete (usually 2-3 minutes)

### 3.2 Get Your Website URL

1. In the Static Web App overview, find the **"URL"** 
2. It will look like: `https://wonderful-field-123456.azurestaticapps.net`
3. Click the URL to test your deployed site

## Step 4: Create Azure Functions API (Integrated Approach)

### 4.1 Create API Folder Structure

Create the following folder structure in your repository:

```
neural-racing-predictor/
├── web-app/
│   └── index.html
├── api/                    # Azure Functions API
│   ├── predict/
│   │   ├── __init__.py
│   │   └── function.json
│   ├── search/
│   │   ├── __init__.py
│   │   └── function.json
│   ├── health/
│   │   ├── __init__.py
│   │   └── function.json
│   ├── requirements.txt    # API dependencies
│   └── host.json
├── simple_api.py          # Original FastAPI (for reference)
└── other files...
```

### 4.2 Convert FastAPI to Azure Functions

Create `api/requirements.txt`:
```text
azure-functions
azure-cosmos
python-dotenv
requests
pandas
numpy
torch
scikit-learn
```

Create `api/host.json`:
```json
{
  "version": "2.0",
  "logging": {
    "applicationInsights": {
      "samplingExcludedTypes": "Request",
      "samplingSettings": {
        "isEnabled": true
      }
    }
  },
  "extensionBundle": {
    "id": "Microsoft.Azure.Functions.ExtensionBundle",
    "version": "[2.*, 3.0.0)"
  },
  "functionTimeout": "00:10:00"
}
```

Create `api/health/function.json`:
```json
{
  "scriptFile": "__init__.py",
  "bindings": [
    {
      "authLevel": "anonymous",
      "type": "httpTrigger",
      "direction": "in",
      "name": "req",
      "methods": ["get"]
    },
    {
      "type": "http",
      "direction": "out",
      "name": "$return"
    }
  ]
}
```

Create `api/health/__init__.py`:
```python
import azure.functions as func
import json
import logging
from datetime import datetime

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Health check endpoint called.')
    
    return func.HttpResponse(
        json.dumps({
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "service": "Neural Racing Predictor API"
        }),
        status_code=200,
        headers={"Content-Type": "application/json"}
    )
```

Create `api/search/function.json`:
```json
{
  "scriptFile": "__init__.py",
  "bindings": [
    {
      "authLevel": "anonymous",
      "type": "httpTrigger",
      "direction": "in",
      "name": "req",
      "methods": ["get"],
      "route": "predictions/search"
    },
    {
      "type": "http",
      "direction": "out",
      "name": "$return"
    }
  ]
}
```

Create `api/search/__init__.py`:
```python
import azure.functions as func
import json
import logging
import os
from cosmos_db_client import CosmosDBClient

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Search predictions endpoint called.')
    
    try:
        # Get query parameters
        date = req.params.get('date')
        venue = req.params.get('venue')
        
        # Initialize Cosmos DB client
        cosmos_client = CosmosDBClient()
        
        # Search predictions
        predictions = cosmos_client.search_predictions(date=date, venue=venue, limit=100)
        
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
            json.dumps({"error": str(e)}),
            status_code=500,
            headers={"Content-Type": "application/json"}
        )
```

### 4.3 Copy Required Modules

Copy these files from your main directory to the `api/` folder:
- `cosmos_db_client.py`
- `racing_com_api_client.py`
- `neural_racing_model.py`
- `feature_creation.py`

### 4.4 Update API Base URL

Edit your `web-app/index.html` file and update the API_BASE_URL:

```javascript
// Find this line in your index.html
const API_BASE_URL = 'http://localhost:8000'; // Update for production

// Replace with relative API path (Azure Functions will handle routing):
const API_BASE_URL = '/api'; // Azure Static Web Apps automatically routes /api to Functions
```

### 4.5 Configure Environment Variables

In Azure Portal, go to your Static Web App → Configuration → Environment variables:

Add these variables:
- `COSMOS_ENDPOINT`: Your Cosmos DB endpoint
- `COSMOS_KEY`: Your Cosmos DB primary key  
- `RACING_API_KEY`: Your racing.com API key

### 4.6 Commit and Deploy

```bash
git add api/ web-app/index.html
git commit -m "Add Azure Functions API integration"
git push origin main
```

The Static Web App will automatically build and deploy both the frontend and API.

## Step 5: Alternative - Separate API Deployment (If Functions Don't Work)

You need to deploy your FastAPI backend separately. Here are the options:

### Option A: Azure Container Apps (Recommended)

1. **Build Docker image**:
   ```bash
   # Create Dockerfile (in project root)
   cat > Dockerfile << 'EOF'
   FROM python:3.9-slim

   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   EXPOSE 8000
   CMD ["python", "simple_api.py"]
   EOF
   ```

2. **Deploy to Azure Container Registry**:
   ```bash
   # Create registry
   az acr create --resource-group neural-racing-rg --name neuralracingacr --sku Basic

   # Build and push image
   az acr build --registry neuralracingacr --image neural-racing-api:latest .
   ```

3. **Create Container App**:
   ```bash
   # Create Container Apps environment
   az containerapp env create \
     --resource-group neural-racing-rg \
     --name neural-racing-env \
     --location eastus2

   # Deploy container
   az containerapp create \
     --resource-group neural-racing-rg \
     --name neural-racing-api \
     --environment neural-racing-env \
     --image neuralracingacr.azurecr.io/neural-racing-api:latest \
     --target-port 8000 \
     --ingress external \
     --env-vars COSMOS_ENDPOINT=$COSMOS_ENDPOINT COSMOS_KEY=$COSMOS_KEY RACING_API_KEY=$RACING_API_KEY
   ```

### Option B: Azure App Service (Simpler)

1. **Create App Service**:
   ```bash
   # Create service plan
   az appservice plan create \
     --resource-group neural-racing-rg \
     --name neural-racing-plan \
     --sku B1 --is-linux

   # Create web app
   az webapp create \
     --resource-group neural-racing-rg \
     --plan neural-racing-plan \
     --name neural-racing-api-app \
     --runtime "PYTHON|3.9"
   ```

2. **Configure environment variables**:
   ```bash
   az webapp config appsettings set \
     --resource-group neural-racing-rg \
     --name neural-racing-api-app \
     --settings \
     COSMOS_ENDPOINT="your-cosmos-endpoint" \
     COSMOS_KEY="your-cosmos-key" \
     RACING_API_KEY="your-racing-api-key"
   ```

3. **Deploy code**:
   ```bash
   # Create deployment package
   zip -r api-deploy.zip simple_api.py cosmos_db_client.py neural_racing_model.py feature_creation.py racing_com_api_client.py neural_racing_predictor.py requirements.txt

   # Deploy
   az webapp deployment source config-zip \
     --resource-group neural-racing-rg \
     --name neural-racing-api-app \
     --src api-deploy.zip
   ```

## Step 6: Configure CORS

Update your `simple_api.py` to allow your Static Web App domain:

```python
# Update CORS middleware in simple_api.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://wonderful-field-123456.azurestaticapps.net",  # Your Static Web App URL
        "http://localhost:3000",  # For local development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

Redeploy your API after making this change.

## Step 7: Test Your Deployment

### 7.1 Test Static Web App

1. Visit your Static Web App URL
2. Check that the page loads correctly
3. Try switching between "Generate New" and "Search Existing" tabs

### 7.2 Test API Connection

1. Open browser developer tools (F12)
2. Try searching for existing predictions
3. Check the Network tab for API calls
4. Look for any CORS or connection errors

### 7.3 Full End-to-End Test

1. Try generating a new prediction (if API supports it)
2. Search for existing predictions
3. Test venue filtering
4. Test expand/collapse functionality

## Step 8: Custom Domain (Optional)

### 8.1 Purchase Domain

Buy a domain from any registrar (GoDaddy, Namecheap, etc.)

### 8.2 Add Custom Domain in Azure

1. In your Static Web App, go to **"Custom domains"**
2. Click **"Add"** → **"Custom domain on other DNS"**
3. Enter your domain (e.g., `racing-predictor.yourdomain.com`)
4. Follow the DNS configuration instructions

### 8.3 Configure DNS

Add the required DNS records at your domain registrar:
- **CNAME record**: Points your subdomain to the Azure Static Web App

## Troubleshooting

### Common Issues:

1. **Static Web App not updating**:
   - Check GitHub Actions in your repository
   - Ensure the workflow file is correct
   - Check for build errors in the Actions tab

2. **API connection errors**:
   - Verify API URL in `index.html`
   - Check CORS configuration
   - Ensure API is deployed and running

3. **GitHub Actions failing**:
   - Check the workflow file in `.github/workflows/`
   - Ensure `app_location: "/web-app"` is correct
   - Check for any build errors in the logs

4. **CORS errors in browser**:
   - Update CORS settings in your API
   - Ensure Static Web App URL is in allowed origins
   - Redeploy the API after CORS changes

### Useful Commands:

```bash
# Check Static Web App status
az staticwebapp show --name neural-racing-predictor --resource-group neural-racing-rg

# View deployment logs
az staticwebapp show --name neural-racing-predictor --resource-group neural-racing-rg --query "repositoryUrl"

# Update environment variables (if using managed functions)
az staticwebapp appsettings set --name neural-racing-predictor --setting-names KEY=VALUE
```

## Monitoring and Maintenance

### Application Insights

1. **Create Application Insights**:
   ```bash
   az monitor app-insights component create \
     --resource-group neural-racing-rg \
     --app neural-racing-insights \
     --location eastus2
   ```

2. **Get connection string** and add to your API configuration

### Regular Maintenance

- Monitor usage and costs in Azure Portal
- Update dependencies regularly
- Review application logs
- Monitor API performance
- Check for security updates

## Cost Estimation

**Azure Static Web Apps:**
- **Free tier**: 100 GB bandwidth/month, 0.5 GB storage
- **Standard tier**: $9/month + usage (custom domains, SLA)

**API Hosting (App Service B1):**
- ~$13/month for basic tier

**Total monthly cost**: $0-22 depending on tier and usage

## Support Resources

- [Azure Static Web Apps Documentation](https://docs.microsoft.com/en-us/azure/static-web-apps/)
- [GitHub Actions for Azure](https://docs.microsoft.com/en-us/azure/developer/github/)
- [FastAPI Deployment Guide](https://fastapi.tiangolo.com/deployment/)

---

Your Neural Racing Predictor web application should now be successfully deployed and accessible worldwide! 🚀