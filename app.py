"""
FastAPI Backend - AI Toxicity Predictor (V4, 6-label multi-label model)
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from predict import ToxicityModerationPipeline

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

app = FastAPI(
    title="AI Toxicity Predictor API",
    description="Multi-label toxicity detection (Jigsaw, DistilBERT V4) with moderation actions",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== MODELS ====================

class PredictionRequest(BaseModel):
    text: str


class BatchRequest(BaseModel):
    texts: List[str]


# ==================== PIPELINE ====================

print("Initializing toxicity pipeline...")
try:
    pipeline = ToxicityModerationPipeline()
    pipeline_ready = True
    print("Pipeline initialized successfully")
except Exception as e:
    pipeline = None
    pipeline_ready = False
    print(f"Pipeline initialization failed: {e}")


def find_label_scores(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return the per-label probabilities from the result."""
    for key in ("class_probabilities", "probabilities", "label_probabilities"):
        value = result.get(key)
        if isinstance(value, dict):
            return value
    return None


def format_result(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "comment": result.get("text"),
        "prediction": result.get("prediction"),
        "toxicity_score": result.get("toxicity_score_percent"),
        "highest_label": result.get("highest_label"),
        "confidence": result.get("confidence_percent"),
        "severity": result.get("severity"),
        "detected_labels": result.get("detected_labels", []),
        "label_scores": find_label_scores(result),
        "action": result.get("action"),
        "reason": result.get("reason"),
        "timestamp": datetime.now().isoformat(),
    }


# ==================== ENDPOINTS ====================

@app.get("/", tags=["Info"])
async def root():
    return {
        "project": "AI Toxicity Predictor",
        "version": "2.0.0",
        "model": "DistilBERT V4 (6-label)",
        "endpoints": ["/health", "/predict", "/predict-batch", "/categories", "/examples", "/docs"],
    }


@app.get("/health", tags=["Health"])
async def health_check():
    if not pipeline_ready:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    device = getattr(pipeline.predictor, "device", "unknown")
    return {"status": "healthy", "model": "DistilBERT-V4-6-label", "device": str(device)}


@app.post("/predict", tags=["Prediction"])
async def predict(request: PredictionRequest):
    if not pipeline_ready:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        result = pipeline.analyze(request.text)
        return format_result(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict-batch", tags=["Prediction"])
async def predict_batch(request: BatchRequest):
    if not pipeline_ready:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    if not request.texts:
        raise HTTPException(status_code=400, detail="Texts list cannot be empty")
    if len(request.texts) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 texts per batch")

    try:
        texts = [t for t in request.texts if t and t.strip()]
        results = [format_result(pipeline.analyze(t)) for t in texts]
        return {
            "total": len(results),
            "predictions": results,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/categories", tags=["Info"])
async def get_categories():
    return {
        "labels": LABELS,
        "note": "Multi-label: a comment can have several labels at once",
    }


@app.get("/examples", tags=["Info"])
async def get_examples():
    return {
        "examples": [
            {"text": "I hope you have a great day.", "expected": "Normal"},
            {"text": "You are stupid and I hate you.", "expected": "Toxic"},
            {"text": "You are a wonderful person.", "expected": "Normal"},
            {"text": "This is a completely unacceptable comment.", "expected": "Normal"},
        ]
    }


# ==================== ERROR HANDLER ====================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat(),
        },
    )


if __name__ == "__main__":
    print("Starting server at http://localhost:8000")
    print("API docs at http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)