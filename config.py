from pathlib import Path
import os


BASE_DIR = Path(__file__).resolve().parent


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "siteaugment-local-development-key")
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024
    UPLOAD_FOLDER = BASE_DIR / "uploads"
    OUTPUT_FOLDER = BASE_DIR / "outputs"
    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
    MAX_IMAGE_PIXELS = 20_000_000
    MAX_GENERATED_IMAGES = 12
