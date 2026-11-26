import numpy as np


def add_noise(image, intensity=0.04, rng=None):
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be an RGB uint8 array")
    if not 0.01 <= float(intensity) <= 0.25:
        raise ValueError("noise intensity must be between 0.01 and 0.25")
    rng = rng or np.random.default_rng()
    sigma = float(intensity) * 40
    noise = rng.normal(0, sigma, image.shape)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
