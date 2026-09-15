"""
FakeNews_AI - FastAPI AI Service

Purpose:
    Exposes the AI Gateway through a REST API.

Architecture:

    Frontend / Backend
            ↓
        POST /predict
            ↓
        AI API
            ↓
        AI Gateway
            ↓
    Hybrid Decision Engine
            ↓
    ML + Evidence + NLI + XAI
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .ai_gateway import predict_claim


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="FakeNews_AI",
    description="Hybrid Real-Time Fake News Detection AI API",
    version="1.0.0"
)


# ============================================================
# REQUEST SCHEMA
# ============================================================

class PredictionRequest(BaseModel):
    claim: str = Field(
        ...,
        min_length=1,
        description="News claim to analyze"
    )


# ============================================================
# HEALTH CHECK
# ============================================================

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


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(request: PredictionRequest):

    claim = request.claim.strip()

    if not claim:
        raise HTTPException(
            status_code=400,
            detail="Claim cannot be empty."
        )

    result = predict_claim(claim)

    return result


# ============================================================
# LOCAL SERVER ENTRY
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "src.ai_api:app",
        host="127.0.0.1",
        port=8000,
        reload=False
    )