from pathlib import Path
import os
import tempfile

from detection.model_config import ONNX_MODEL_FILENAME


BASE_DIR = Path(__file__).resolve().parent


def default_data_dir():
    configured_dir = os.environ.get("SITEAUGMENT_DATA_DIR")
    if configured_dir:
        return Path(configured_dir)
    if os.environ.get("VERCEL") or os.environ.get("VERCEL_ENV"):
        return Path(tempfile.gettempdir()) / "siteaugment-ai"
    return BASE_DIR


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "siteaugment-local-development-key")
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024
    DATA_DIR = default_data_dir()
    UPLOAD_FOLDER = DATA_DIR / "uploads"
    OUTPUT_FOLDER = DATA_DIR / "outputs"
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
