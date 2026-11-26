from pathlib import Path
import os

from detection.model_config import ONNX_MODEL_FILENAME


BASE_DIR = Path(__file__).resolve().parent


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "siteaugment-local-development-key")
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024
    UPLOAD_FOLDER = BASE_DIR / "uploads"
    OUTPUT_FOLDER = BASE_DIR / "outputs"
    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
    MAX_IMAGE_PIXELS = 20_000_000
    MAX_GENERATED_IMAGES = 12
    PPE_MODEL_PATH = os.environ.get("PPE_MODEL_PATH") or str(
        BASE_DIR / "models" / ONNX_MODEL_FILENAME
    )
    PPE_MODEL_REPO_ID = "Tanishjain9/yolov8n-ppe-detection-6classes"
    PPE_MODEL_REVISION = "a2c1a97646ade96e48adda74781dd455d0456c40"
    PPE_MODEL_FILENAME = "best.pt"
    PPE_DEFAULT_CONFIDENCE = 0.25
