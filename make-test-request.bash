gcloud auth application-default login
echo {
    "instances": [{
        "image_bytes": {
            "b64": “<BASE64-OF-IMAGE-HERE>”
        }
    }],
    "signature_name": "serving_image_b64string"
} >> sample_request.json
ENDPOINT_ID="<Your-Endpoint-ID-Here>"
PROJECT_ID="<Your-Project-ID-Here>"
INPUT_DATA_FILE="sample_request.json"
curl \
    -X POST \
    -H "Authorization: Bearer $(gcloud auth print-access-token)" \
    -H "Content-Type: application/json" \
    https://us-central1-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/us-central1/endpoints/${ENDPOINT_ID}:rawPredict \
    -d "@${INPUT_DATA_FILE}"