import logging
import math
import threading

import cv2
import numpy as np

from .model_config import (
    DEFAULT_CONFIDENCE,
    MAX_CONFIDENCE,
    MIN_CONFIDENCE,
    MODEL_FILENAME,
    MODEL_REPO_ID,
    MODEL_REVISION,
)


logger = logging.getLogger(__name__)


class DetectorUnavailable(RuntimeError):
    """Raised when the model cannot be loaded or used."""


class PPEDetector:
    """Lazy-loaded YOLO PPE detector with validated, JSON-ready results."""

    def __init__(
        self,
        model_path=None,
        repo_id=MODEL_REPO_ID,
        revision=MODEL_REVISION,
        filename=MODEL_FILENAME,
        model_loader=None,
    ):
        self.model_path = model_path
        self.repo_id = repo_id
        self.revision = revision
        self.filename = filename
        self.model_loader = model_loader
        self._model = None
        self._load_error = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    @staticmethod
    def validate_confidence(confidence):
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError("confidence must be a number")
        try:
            confidence = float(confidence)
        except OverflowError as exc:
            raise ValueError("confidence must be a finite number") from exc
        if not math.isfinite(confidence) or not MIN_CONFIDENCE <= confidence <= MAX_CONFIDENCE:
            raise ValueError(
                f"confidence must be between {MIN_CONFIDENCE} and {MAX_CONFIDENCE}"
            )
        return confidence

    @staticmethod
    def validate_image(image):
        if (
            not isinstance(image, np.ndarray)
            or image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[2] != 3
            or image.shape[0] <= 0
            or image.shape[1] <= 0
        ):
            raise ValueError("image must be a non-empty RGB uint8 array")

    def _load_model(self):
        if self.model_loader is not None:
            return self.model_loader(self.model_path)

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise DetectorUnavailable(
                "PPE detection dependencies are not installed."
            ) from exc

        if self.model_path:
            checkpoint = self.model_path
        else:
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as exc:
                raise DetectorUnavailable(
                    "PPE detection dependencies are not installed."
                ) from exc
            checkpoint = hf_hub_download(
                repo_id=self.repo_id,
                filename=self.filename,
                revision=self.revision,
            )

        return YOLO(checkpoint)

    def _get_model(self):
        if self._model is not None:
            return self._model
        if self._load_error is not None:
            raise DetectorUnavailable(
                "PPE detection model is unavailable. Check model setup and server logs."
            ) from self._load_error
        with self._load_lock:
            if self._model is not None:
                return self._model
            if self._load_error is not None:
                raise DetectorUnavailable(
                    "PPE detection model is unavailable. Check model setup and server logs."
                ) from self._load_error
            try:
                self._model = self._load_model()
            except Exception as exc:
                self._load_error = exc
                logger.exception("Could not load the PPE detection model")
                raise DetectorUnavailable(
                    "PPE detection model is unavailable. Check model setup and server logs."
                ) from exc
        return self._model

    @staticmethod
    def _as_numpy(values):
        if hasattr(values, "cpu"):
            values = values.cpu()
        if hasattr(values, "numpy"):
            values = values.numpy()
        return np.asarray(values)

    @staticmethod
    def _class_names(names):
        if isinstance(names, dict):
            return {int(key): str(value) for key, value in names.items()}
        if isinstance(names, (list, tuple)):
            return {index: str(value) for index, value in enumerate(names)}
        raise DetectorUnavailable("PPE detection model has invalid class metadata.")

    def detect(self, image, confidence=DEFAULT_CONFIDENCE):
        self.validate_image(image)
        confidence = self.validate_confidence(confidence)
        model = self._get_model()
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        try:
            with self._inference_lock:
                predictions = model.predict(
                    source=bgr_image,
                    conf=confidence,
                    verbose=False,
                )
        except Exception as exc:
            logger.exception("PPE inference failed")
            raise DetectorUnavailable("PPE detection failed. Check server logs.") from exc

        if not predictions:
            raise DetectorUnavailable("PPE detection model returned no prediction result.")
        result = predictions[0]
        names = self._class_names(
            getattr(result, "names", None) or getattr(model, "names", None)
        )
        boxes = getattr(result, "boxes", None)
        detections = []
        if boxes is not None and len(boxes):
            coordinates = self._as_numpy(boxes.xyxy)
            class_ids = self._as_numpy(boxes.cls).reshape(-1)
            scores = self._as_numpy(boxes.conf).reshape(-1)
            if not (len(coordinates) == len(class_ids) == len(scores)):
                raise DetectorUnavailable("PPE detection model returned invalid boxes.")
            if coordinates.ndim != 2 or coordinates.shape[1] != 4:
                raise DetectorUnavailable("PPE detection model returned invalid boxes.")
            image_height, image_width = image.shape[:2]
            for coordinates_row, raw_class_id, raw_score in zip(coordinates, class_ids, scores):
                raw_class_id = float(raw_class_id)
                score = float(raw_score)
                if (
                    not math.isfinite(raw_class_id)
                    or not raw_class_id.is_integer()
                    or not math.isfinite(score)
                    or score < confidence
                    or score > 1
                ):
                    continue
                class_id = int(raw_class_id)
                if class_id not in names:
                    continue
                if len(coordinates_row) != 4 or not np.isfinite(coordinates_row).all():
                    continue
                x1, y1, x2, y2 = (float(value) for value in coordinates_row)
                x1 = min(max(x1, 0.0), float(image_width))
                x2 = min(max(x2, 0.0), float(image_width))
                y1 = min(max(y1, 0.0), float(image_height))
                y2 = min(max(y2, 0.0), float(image_height))
                if x2 <= x1 or y2 <= y1:
                    continue
                detections.append({
                    "class_id": class_id,
                    "label": names[class_id],
                    "confidence": round(score, 6),
                    "bbox": {
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2),
                    },
                })

        class_counts = {label: 0 for label in names.values()}
        for detection in detections:
            class_counts[detection["label"]] += 1
        scores = [item["confidence"] for item in detections]
        summary = {
            "total_detections": len(detections),
            "class_counts": class_counts,
            "confidence": {
                "min": round(min(scores), 6) if scores else None,
                "max": round(max(scores), 6) if scores else None,
                "mean": round(sum(scores) / len(scores), 6) if scores else None,
            },
        }
        return {
            "detections": detections,
            "summary": summary,
            "classes": [names[key] for key in sorted(names)],
        }

    @staticmethod
    def visualize(image, detections):
        PPEDetector.validate_image(image)
        if not isinstance(detections, list):
            raise ValueError("detections must be a list")
        output = image.copy()
        palette = [
            (42, 190, 219),
            (142, 117, 244),
            (87, 190, 133),
            (236, 164, 83),
            (228, 107, 154),
            (91, 144, 239),
            (207, 204, 100),
        ]
        height, width = output.shape[:2]
        thickness = max(1, round(min(height, width) / 300))
        font_scale = max(0.4, min(height, width) / 900)
        for item in detections:
            if not isinstance(item, dict) or not isinstance(item.get("bbox"), dict):
                raise ValueError("each detection must contain a bounding box")
            box = item["bbox"]
            x1, y1, x2, y2 = (
                int(round(float(box[key]))) for key in ("x1", "y1", "x2", "y2")
            )
            if x1 < 0 or y1 < 0 or x2 > width or y2 > height or x2 <= x1 or y2 <= y1:
                raise ValueError("detection bounding box is outside the image")
            label = str(item.get("label", "Object"))
            confidence = float(item.get("confidence", 0))
            if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError("detection confidence must be between 0 and 1")
            color = palette[int(item.get("class_id", 0)) % len(palette)]
            cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
            caption = f"{label} {confidence:.0%}"
            (text_width, text_height), baseline = cv2.getTextSize(
                caption, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            text_y = max(text_height + 4, y1)
            cv2.rectangle(
                output,
                (x1, text_y - text_height - 4),
                (min(width - 1, x1 + text_width + 6), text_y + baseline - 1),
                color,
                -1,
            )
            cv2.putText(
                output,
                caption,
                (x1 + 3, text_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (10, 14, 24),
                thickness,
                cv2.LINE_AA,
            )
        return output
