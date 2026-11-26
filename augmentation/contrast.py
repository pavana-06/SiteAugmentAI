import numpy as np


def adjust_contrast(image, factor=1.1):
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be an RGB uint8 array")
    if not 0.65 <= float(factor) <= 1.5:
        raise ValueError("contrast factor must be between 0.65 and 1.5")
    midpoint = 127.5
    adjusted = (image.astype(np.float32) - midpoint) * float(factor) + midpoint
    return np.clip(adjusted, 0, 255).astype(np.uint8)
