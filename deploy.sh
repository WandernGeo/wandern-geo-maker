#!/bin/bash
# Deploy Geo Echo Maker Cloud Function

set -e

PROJECT_ID="wandern-project-startup"
REGION="us-central1"
FUNCTION_NAME="geo-maker"

echo "🚀 Deploying $FUNCTION_NAME..."

gcloud functions deploy $FUNCTION_NAME \
    --gen2 \
    --runtime=python311 \
    --region=$REGION \
    --source=. \
    --entry-point=process_echo \
    --trigger-http \
    --allow-unauthenticated \
    --memory=1GB \
    --timeout=120s \
    --set-env-vars="GCS_BUCKET=wandern-geo-echoes" \
    --project=$PROJECT_ID

echo "✅ Deploy complete!"
echo "URL: https://$REGION-$PROJECT_ID.cloudfunctions.net/$FUNCTION_NAME"
