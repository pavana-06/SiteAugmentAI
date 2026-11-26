import cv2
import numpy as np


def adjust_contrast(image, factor=1.2):
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be an RGB uint8 array")
    if not 0.1 <= float(factor) <= 3.0:
        raise ValueError("contrast factor must be between 0.1 and 3")
    return cv2.convertScaleAbs(image, alpha=float(factor), beta=0)
