```python
"""
FakeNews_AI - FastAPI AI Service

This file handles API requests and sends news claims
to the AI Gateway for prediction.

Available endpoints:
    /       - Check if the API is running
    /health - Check service status
    /predict - Analyze a news claim
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .ai_gateway import predict_claim


app = FastAPI(
    title="FakeNews_AI",
    description="Hybrid Real-Time Fake News Detection API",
    version="1.0.0"
)


# Request format for news prediction
class PredictionRequest(BaseModel):
    claim: str = Field(
        ...,
        min_length=1,
        description="News claim to analyze"
    )


@app.get("/")
def root():
    return {
        "service": "FakeNews_AI",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "FakeNews_AI"
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    # Get the claim and remove extra spaces.
    claim = request.claim.strip()

    if not claim:
        raise HTTPException(
            status_code=400,
            detail="Claim cannot be empty."
        )

    # Get the result from the AI Gateway.
    result = predict_claim(claim)

    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.ai_api:app",
        host="127.0.0.1",
        port=8000,
        reload=False
    )
```
