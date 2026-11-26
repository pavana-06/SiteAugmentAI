import cv2
import numpy as np


def add_shadow(image, intensity=0.12, rng=None):
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be an RGB uint8 array")
    if not 0.05 <= float(intensity) <= 0.6:
        raise ValueError("shadow intensity must be between 0.05 and 0.6")
    rng = rng or np.random.default_rng()
    height, width = image.shape[:2]
    points = np.array([
        [rng.integers(0, width), 0],
        [rng.integers(0, width), 0],
        [rng.integers(0, width), height - 1],
        [rng.integers(0, width), height - 1],
    ], dtype=np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(3, min(height, width) * 0.06))
    shade = 1.0 - float(intensity) * (mask.astype(np.float32) / 255.0)
    return np.clip(image.astype(np.float32) * shade[:, :, None], 0, 255).astype(np.uint8)
