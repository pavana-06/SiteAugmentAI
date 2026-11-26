import cv2
import numpy as np


def _validate(image):
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be an RGB uint8 array")


def gaussian_blur(image, strength=0.12):
    _validate(image)
    if not 0.05 <= float(strength) <= 0.6:
        raise ValueError("blur strength must be between 0.05 and 0.6")
    sigma = 0.25 + float(strength) * 1.25
    return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)


def motion_blur(image, strength=0.1):
    _validate(image)
    if not 0.05 <= float(strength) <= 0.6:
        raise ValueError("blur strength must be between 0.05 and 0.6")
    normalized = (float(strength) - 0.05) / 0.55
    size = 3 + 2 * round(normalized * 2)
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[size // 2, :] = 1.0 / size
    blurred = cv2.filter2D(image, -1, kernel)
    blend = 0.12 + 0.55 * normalized
    return cv2.addWeighted(image, 1.0 - blend, blurred, blend, 0)
