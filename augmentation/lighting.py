import cv2
import numpy as np


def _validate_image(image):
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be an RGB uint8 array")


def brightness(image, amount=0.1):
    _validate_image(image)
    if not -0.5 <= float(amount) <= 0.5:
        raise ValueError("brightness amount must be between -0.5 and 0.5")
    return np.clip(image.astype(np.float32) + float(amount) * 255, 0, 255).astype(np.uint8)


def increase_brightness(image, amount=0.1):
    if not 0 < float(amount) <= 0.5:
        raise ValueError("brightness amount must be between 0 and 0.5")
    return brightness(image, amount)


def decrease_brightness(image, amount=0.1):
    if not 0 < float(amount) <= 0.5:
        raise ValueError("brightness amount must be between 0 and 0.5")
    return brightness(image, -amount)


def random_brightness(image, intensity=0.12, rng=None):
    if not 0 <= float(intensity) <= 0.4:
        raise ValueError("intensity must be between 0 and 0.4")
    rng = rng or np.random.default_rng()
    return brightness(image, rng.uniform(-intensity, intensity))


def gamma_correction(image, gamma=1.1):
    _validate_image(image)
    if not 0.65 <= float(gamma) <= 1.8:
        raise ValueError("gamma must be between 0.65 and 1.8")
    lookup = np.clip(((np.arange(256) / 255.0) ** (1.0 / float(gamma))) * 255, 0, 255).astype(np.uint8)
    return cv2.LUT(image, lookup)
