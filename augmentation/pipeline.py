import numpy as np
import math

from .blur import gaussian_blur, motion_blur
from .contrast import adjust_contrast
from .haze import add_haze
from .lighting import brightness, gamma_correction, random_brightness
from .noise import add_noise
from .occlusion import add_occlusion
from .shadows import add_shadow


SUPPORTED_AUGMENTATIONS = {
    "brightness", "contrast", "gamma", "shadow", "gaussian_blur",
    "motion_blur", "noise", "occlusion", "haze",
}


def _validated_number(value, name, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    value = float(value)
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def apply_augmentations(image, augmentations, seed=None):
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be an RGB uint8 array")
    if not isinstance(augmentations, list) or not augmentations:
        raise ValueError("select at least one augmentation")
    rng = np.random.default_rng(seed)
    result = image.copy()
    labels = []
    for item in augmentations:
        if not isinstance(item, dict):
            raise ValueError("each augmentation must be an object")
        name = item.get("name")
        params = item.get("params", {})
        if name not in SUPPORTED_AUGMENTATIONS or not isinstance(params, dict):
            raise ValueError("unsupported augmentation or invalid parameters")
        if name == "brightness":
            intensity = _validated_number(params.get("intensity", 0.25), "brightness intensity", 0.0, 1.0)
            result = random_brightness(result, intensity, rng)
        elif name == "contrast":
            factor = _validated_number(params.get("factor", 1.35), "contrast factor", 0.1, 3.0)
            result = adjust_contrast(result, factor)
        elif name == "gamma":
            gamma = _validated_number(params.get("gamma", 1.4), "gamma", 0.1, 5.0)
            result = gamma_correction(result, gamma)
        elif name == "shadow":
            intensity = _validated_number(params.get("intensity", 0.35), "shadow intensity", 0.05, 0.85)
            result = add_shadow(result, intensity, rng)
        elif name == "gaussian_blur":
            strength = _validated_number(params.get("strength", 0.35), "blur strength", 0.05, 1.0)
            result = gaussian_blur(result, strength)
        elif name == "motion_blur":
            strength = _validated_number(params.get("strength", 0.35), "blur strength", 0.05, 1.0)
            result = motion_blur(result, strength)
        elif name == "noise":
            intensity = _validated_number(params.get("intensity", 0.12), "noise intensity", 0.01, 0.5)
            result = add_noise(result, intensity, rng)
        elif name == "occlusion":
            intensity = _validated_number(params.get("intensity", 0.2), "occlusion intensity", 0.03, 0.4)
            result = add_occlusion(result, intensity, rng)
        elif name == "haze":
            intensity = _validated_number(params.get("intensity", 0.25), "haze intensity", 0.02, 0.65)
            result = add_haze(result, intensity, rng)
        labels.append(name.replace("_", " ").title())
    return result, labels
