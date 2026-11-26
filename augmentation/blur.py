import cv2
import numpy as np


def _validate(image):
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be an RGB uint8 array")


def gaussian_blur(image, strength=0.35):
    _validate(image)
    if not 0.05 <= float(strength) <= 1.0:
        raise ValueError("blur strength must be between 0.05 and 1")
    kernel_size = max(3, int(round(3 + float(strength) * 14)) | 1)
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def motion_blur(image, strength=0.35):
    _validate(image)
    if not 0.05 <= float(strength) <= 1.0:
        raise ValueError("blur strength must be between 0.05 and 1")
    size = max(3, int(round(3 + float(strength) * 18)) | 1)
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[size // 2, :] = 1.0 / size
    return cv2.filter2D(image, -1, kernel)
