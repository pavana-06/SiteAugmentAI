import cv2
import numpy as np


def add_occlusion(image, intensity=0.2, rng=None):
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be an RGB uint8 array")
    if not 0.03 <= float(intensity) <= 0.4:
        raise ValueError("occlusion intensity must be between 0.03 and 0.4")
    rng = rng or np.random.default_rng()
    height, width = image.shape[:2]
    fraction = min(0.45, 0.12 + float(intensity) * 0.7)
    box_width = max(1, int(width * np.sqrt(fraction) * rng.uniform(0.7, 1.2)))
    box_height = max(1, min(int(height * 0.65), int(height * fraction * width / box_width)))
    box_width = min(box_width, max(1, int(width * 0.65)))
    x = int(rng.integers(0, max(1, width - box_width + 1)))
    y = int(rng.integers(0, max(1, height - box_height + 1)))
    output = image.copy()
    color = rng.integers(20, 90, size=3, dtype=np.uint8)
    overlay = output.copy()
    cv2.rectangle(overlay, (x, y), (x + box_width, y + box_height), tuple(int(v) for v in color), -1)
    alpha = min(0.9, 0.45 + float(intensity))
    output[y:y + box_height, x:x + box_width] = (
        output[y:y + box_height, x:x + box_width].astype(np.float32) * (1 - alpha)
        + overlay[y:y + box_height, x:x + box_width].astype(np.float32) * alpha
    ).astype(np.uint8)
    return output
