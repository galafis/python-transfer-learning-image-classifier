"""Tests for Transfer Learning model factory and inference."""

import pytest
import numpy as np


class TestModelFactory:
    """Tests for model creation and configuration."""

    def test_model_registry_has_all_models(self):
        """Test that all expected models are in the registry."""
        from src.models.model_factory import MODEL_REGISTRY
        assert "vgg16" in MODEL_REGISTRY
        assert "resnet50" in MODEL_REGISTRY
        assert "efficientnetb0" in MODEL_REGISTRY

    def test_create_model_invalid_name(self):
        """Test that invalid model name raises ValueError."""
        from src.models.model_factory import create_model
        with pytest.raises(ValueError, match="not supported"):
            create_model("invalid_model", num_classes=10)

    def test_create_model_resnet50(self):
        """Test ResNet50 model creation."""
        from src.models.model_factory import create_model
        model = create_model("resnet50", num_classes=5)
        assert model is not None
        assert model.name == "tl_resnet50"
        output_shape = model.output_shape
        assert output_shape[-1] == 5

    def test_create_model_binary(self):
        """Test binary classification model."""
        from src.models.model_factory import create_model
        model = create_model("resnet50", num_classes=2)
        assert model is not None
        output_shape = model.output_shape
        assert output_shape[-1] == 1

    def test_model_input_shape(self):
        """Test custom input shape."""
        from src.models.model_factory import create_model
        model = create_model(
            "resnet50", num_classes=3, input_shape=(128, 128, 3)
        )
        assert model.input_shape == (None, 128, 128, 3)


class TestPreprocessing:
    """Tests for image preprocessing."""

    def test_preprocess_image(self):
        """Test image preprocessing output shape."""
        from PIL import Image
        from src.inference.api import preprocess_image

        img = Image.new("RGB", (300, 300), color="red")
        result = preprocess_image(img)
        assert result.shape == (1, 224, 224, 3)
        assert result.max() <= 1.0
        assert result.min() >= 0.0

    def test_preprocess_grayscale(self):
        """Test grayscale image conversion."""
        from PIL import Image
        from src.inference.api import preprocess_image

        img = Image.new("L", (100, 100), color=128)
        result = preprocess_image(img)
        assert result.shape == (1, 224, 224, 3)


class TestAPI:
    """Tests for FastAPI endpoints."""

    def test_health_endpoint(self):
        """Test health check returns valid response."""
        from fastapi.testclient import TestClient
        from src.inference.api import app

        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_predict_no_model(self):
        """Test prediction fails gracefully without model."""
        from fastapi.testclient import TestClient
        from src.inference.api import app, _model
        import src.inference.api as api_module

        api_module._model = None
        client = TestClient(app)

        from io import BytesIO
        from PIL import Image
        img = Image.new("RGB", (100, 100), color="blue")
        buf = BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        response = client.post(
            "/predict",
            files={"file": ("test.png", buf, "image/png")},
        )
        assert response.status_code == 503
