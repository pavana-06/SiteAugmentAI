import cv2
import numpy as np


def add_haze(image, intensity=0.25, rng=None):
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be an RGB uint8 array")
    if not 0.02 <= float(intensity) <= 0.65:
        raise ValueError("haze intensity must be between 0.02 and 0.65")
    rng = rng or np.random.default_rng()
    height, width = image.shape[:2]
    veil = rng.uniform(0.75, 0.95, size=(max(2, height // 24), max(2, width // 24))).astype(np.float32)
    veil = cv2.resize(veil, (width, height), interpolation=cv2.INTER_CUBIC)
    veil = cv2.GaussianBlur(veil, (0, 0), sigmaX=max(1, min(height, width) * 0.04))
    strength = float(intensity)
    transmission = np.clip(1.0 - strength * (0.65 + 0.35 * veil), 0.25, 1.0)
    airlight = np.array([221, 226, 229], dtype=np.float32)
    result = image.astype(np.float32) * transmission[:, :, None] + airlight * (1 - transmission[:, :, None])
    return np.clip(result, 0, 255).astype(np.uint8)
