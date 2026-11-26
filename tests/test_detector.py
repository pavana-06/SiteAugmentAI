import builtins
from pathlib import Path

import numpy as np
import pytest

from detection.detector import DetectorUnavailable, PPEDetector


class FakeBoxes:
    def __init__(self):
        self.xyxy = np.array([
            [-4.0, 5.0, 50.0, 80.0],
            [12.0, 14.0, 40.0, 60.0],
        ])
        self.cls = np.array([3, 0])
        self.conf = np.array([0.91, 0.31])

    def __len__(self):
        return len(self.cls)


class FakeResult:
    names = {
        0: "Gloves",
        1: "Vest",
        2: "goggles",
        3: "helmet",
        4: "mask",
        5: "safety_shoe",
    }
    boxes = FakeBoxes()


class FakeModel:
    names = FakeResult.names

    def __init__(self):
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return [FakeResult()]


class FakeOrtInput:
    name = "images"


class FakeOrtSession:
    def __init__(self):
        self.tensor = None

    def get_inputs(self):
        return [FakeOrtInput()]

    def run(self, _outputs, inputs):
        self.tensor = inputs["images"]
        predictions = np.zeros((1, 10, 3), dtype=np.float32)
        predictions[0, :4, :] = np.array([
            [160, 160, 160],
            [230, 230, 230],
            [200, 200, 200],
            [200, 200, 200],
        ])
        predictions[0, 4 + 3, :] = [0.91, 0.80, 0.10]
        predictions[0, 4 + 1, 2] = 0.70
        return [predictions]


@pytest.fixture
def image():
    return np.full((96, 128, 3), 110, dtype=np.uint8)


def test_model_initializes_lazily_once_and_returns_detection_structure(image):
    models = []

    def load_model(_path):
        model = FakeModel()
        models.append(model)
        return model

    detector = PPEDetector(model_loader=load_model)
    first = detector.detect(image, 0.25)
    second = detector.detect(image, 0.25)

    assert len(models) == 1
    assert first["classes"] == ["Gloves", "Vest", "goggles", "helmet", "mask", "safety_shoe"]
    assert first["summary"]["total_detections"] == 2
    assert first["summary"]["class_counts"]["helmet"] == 1
    assert first["summary"]["class_counts"]["Gloves"] == 1
    assert first["summary"]["confidence"]["mean"] == 0.61
    assert first["detections"] == second["detections"]
    assert first["detections"][0]["bbox"] == {"x1": 0.0, "y1": 5.0, "x2": 50.0, "y2": 80.0}


def test_confidence_threshold_filters_detections(image):
    model = FakeModel()
    detector = PPEDetector(model_loader=lambda _path: model)

    result = detector.detect(image, 0.5)

    assert result["summary"]["total_detections"] == 1
    assert result["detections"][0]["label"] == "helmet"
    assert model.calls[-1]["conf"] == 0.5


@pytest.mark.parametrize("confidence", [True, 0, 1, float("nan"), float("inf"), "nope", 10**400])
def test_invalid_confidence_is_rejected(image, confidence):
    detector = PPEDetector(model_loader=lambda _path: FakeModel())
    with pytest.raises(ValueError):
        detector.detect(image, confidence)


def test_invalid_image_is_rejected_without_loading_model():
    detector = PPEDetector(model_loader=lambda _path: FakeModel())
    with pytest.raises(ValueError):
        detector.detect(np.zeros((10, 10), dtype=np.uint8))


def test_model_load_errors_are_safe_and_cached(image):
    attempts = []

    def fail_load(_path):
        attempts.append(True)
        raise OSError("private checkpoint path")

    detector = PPEDetector(model_loader=fail_load)
    for _ in range(2):
        with pytest.raises(DetectorUnavailable, match="PPE detection model is unavailable") as error:
            detector.detect(image)
        assert "private checkpoint path" not in str(error.value)
        assert "ONNX Runtime" in str(error.value)
    assert len(attempts) == 1


def test_missing_optional_inference_dependency_returns_clear_error(image, monkeypatch):
    original_import = builtins.__import__

    def import_without_ultralytics(name, *args, **kwargs):
        if name == "ultralytics":
            raise ImportError("optional inference package is absent")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_ultralytics)
    detector = PPEDetector()

    with pytest.raises(DetectorUnavailable, match="PPE detection model is unavailable"):
        detector.detect(image)


def test_onnx_inference_letterboxes_maps_boxes_and_applies_classwise_nms(image):
    session = FakeOrtSession()
    detector = PPEDetector(
        model_path="ppe.onnx",
        model_loader=lambda _path: session,
    )

    result = detector.detect(image, 0.25)

    assert session.tensor.shape == (1, 3, 640, 640)
    assert result["classes"] == ["Gloves", "Vest", "goggles", "helmet", "mask", "safety_shoe"]
    assert result["summary"]["total_detections"] == 2
    assert result["summary"]["class_counts"]["helmet"] == 1
    assert result["summary"]["class_counts"]["Vest"] == 1
    assert result["detections"][0]["bbox"] == {
        "x1": 12.0,
        "y1": 10.0,
        "x2": 52.0,
        "y2": 50.0,
    }


def test_onnx_inference_respects_confidence_threshold(image):
    detector = PPEDetector(
        model_path="ppe.onnx",
        model_loader=lambda _path: FakeOrtSession(),
    )

    result = detector.detect(image, 0.85)

    assert result["summary"]["total_detections"] == 1
    assert result["detections"][0]["label"] == "helmet"


def test_bundled_onnx_model_runs_real_cpu_inference():
    model_path = Path(__file__).resolve().parents[1] / "models" / "ppe-yolov8n.onnx"
    detector = PPEDetector(model_path=model_path)
    image = np.zeros((96, 128, 3), dtype=np.uint8)

    result = detector.detect(image, 0.25)

    assert result["classes"] == ["Gloves", "Vest", "goggles", "helmet", "mask", "safety_shoe"]
    assert result["summary"]["total_detections"] == len(result["detections"])
    for detection in result["detections"]:
        box = detection["bbox"]
        assert 0 <= box["x1"] < box["x2"] <= image.shape[1]
        assert 0 <= box["y1"] < box["y2"] <= image.shape[0]
        assert 0.25 <= detection["confidence"] <= 1


def test_visualization_draws_boxes_on_copy_and_validates_bounds(image):
    original = image.copy()
    detections = [{
        "class_id": 3,
        "label": "helmet",
        "confidence": 0.91,
        "bbox": {"x1": 5, "y1": 5, "x2": 70, "y2": 70},
    }]
    annotated = PPEDetector.visualize(image, detections)

    assert annotated.shape == image.shape
    assert np.array_equal(image, original)
    assert np.any(annotated != image)
    with pytest.raises(ValueError, match="outside the image"):
        PPEDetector.visualize(image, [{
            **detections[0],
            "bbox": {"x1": -1, "y1": 5, "x2": 70, "y2": 70},
        }])


def test_visualization_rejects_invalid_confidence(image):
    with pytest.raises(ValueError, match="confidence"):
        PPEDetector.visualize(image, [{
            "class_id": 0,
            "label": "helmet",
            "confidence": 1.5,
            "bbox": {"x1": 1, "y1": 1, "x2": 10, "y2": 10},
        }])
