from io import BytesIO
import math

import numpy as np
from PIL import Image, UnidentifiedImageError


def decode_image(data, allowed_extensions, max_pixels):
    try:
        with Image.open(BytesIO(data)) as source:
            extension = source.format.lower() if source.format else ""
            if extension not in allowed_extensions:
                raise ValueError("Only JPG, JPEG, and PNG images are supported.")
            if source.width * source.height > max_pixels:
                raise ValueError("Image dimensions exceed the allowed limit.")
            image = source.convert("RGB")
            return np.asarray(image, dtype=np.uint8).copy()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("The uploaded file is not a valid image.") from exc


def encode_png(image):
    buffer = BytesIO()
    Image.fromarray(image, mode="RGB").save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def image_metrics(original, augmented):
    if original.shape != augmented.shape:
        raise ValueError("images must have matching dimensions")
    first = original.astype(np.float64)
    second = augmented.astype(np.float64)
    mse = float(np.mean((first - second) ** 2))
    psnr = float("inf") if mse == 0 else 10 * math.log10((255.0 ** 2) / mse)
    mean_a = float(first.mean())
    mean_b = float(second.mean())
    var_a = float(first.var())
    var_b = float(second.var())
    covariance = float(np.mean((first - mean_a) * (second - mean_b)))
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    denominator = (mean_a ** 2 + mean_b ** 2 + c1) * (var_a + var_b + c2)
    ssim = ((2 * mean_a * mean_b + c1) * (2 * covariance + c2)) / denominator if denominator else 1.0
    return {
        "width": int(original.shape[1]),
        "height": int(original.shape[0]),
        "mean_brightness": round(float(augmented.mean()), 2),
        "psnr": None if math.isinf(psnr) else round(psnr, 2),
        "ssim": round(float(np.clip(ssim, -1, 1)), 4),
    }
