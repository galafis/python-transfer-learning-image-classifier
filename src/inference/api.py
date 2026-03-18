"""FastAPI inference endpoint for Transfer Learning classifier."""

import io
import logging
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Transfer Learning Image Classifier API",
    description="Classify images using fine-tuned deep learning models",
    version="1.0.0",
)

# Global model holder
_model = None
_class_names: List[str] = []


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool


def load_model(model_path: str = "models/best_model.keras") -> None:
    """Load a trained model for inference."""
    global _model, _class_names
    try:
        import tensorflow as tf
        _model = tf.keras.models.load_model(model_path)
        logger.info(f"Model loaded from {model_path}")
    except Exception as e:
        logger.warning(f"Could not load model: {e}")


def preprocess_image(
    image: Image.Image,
    target_size: tuple = (224, 224),
) -> np.ndarray:
    """Preprocess an image for model inference."""
    image = image.convert("RGB")
    image = image.resize(target_size)
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=_model is not None,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """Classify an uploaded image."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        processed = preprocess_image(image)

        predictions = _model.predict(processed, verbose=0)
        predicted_idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_idx])

        class_name = (
            _class_names[predicted_idx]
            if _class_names
            else f"class_{predicted_idx}"
        )

        probabilities = {
            (_class_names[i] if i < len(_class_names) else f"class_{i}"): float(p)
            for i, p in enumerate(predictions[0])
        }

        return PredictionResponse(
            predicted_class=class_name,
            confidence=confidence,
            probabilities=probabilities,
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    load_model()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
