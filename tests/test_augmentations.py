import numpy as np
import pytest

from augmentation.blur import gaussian_blur, motion_blur
from augmentation.contrast import adjust_contrast
from augmentation.haze import add_haze
from augmentation.lighting import brightness, decrease_brightness, gamma_correction, increase_brightness
from augmentation.noise import add_noise
from augmentation.occlusion import add_occlusion
from augmentation.pipeline import apply_augmentations
from augmentation.shadows import add_shadow
from utils.image_utils import image_metrics


@pytest.fixture
def image():
    rows, cols = np.indices((96, 128))
    values = ((rows * 3 + cols * 2) % 180 + 35).astype(np.uint8)
    return np.dstack((values, values // 2 + 20, values // 3 + 30))


@pytest.mark.parametrize("operation", [
    lambda image: brightness(image, 0.25),
    lambda image: brightness(image, -0.25),
    lambda image: gamma_correction(image, 1.7),
    lambda image: adjust_contrast(image, 1.5),
    lambda image: add_shadow(image, 0.5, np.random.default_rng(4)),
    lambda image: gaussian_blur(image, 0.6),
    lambda image: motion_blur(image, 0.6),
    lambda image: add_noise(image, 0.15, np.random.default_rng(4)),
    lambda image: add_occlusion(image, 0.25, np.random.default_rng(4)),
    lambda image: add_haze(image, 0.3, np.random.default_rng(4)),
])
def test_augmentation_changes_image_and_preserves_shape(image, operation):
    result = operation(image)
    assert result.dtype == np.uint8
    assert result.shape == image.shape
    assert np.any(result != image)


def test_pipeline_combines_transforms_reproducibly(image):
    selected = [
        {"name": "brightness", "params": {"intensity": 0.25}},
        {"name": "noise", "params": {"intensity": 0.1}},
        {"name": "haze", "params": {"intensity": 0.2}},
    ]
    first, labels = apply_augmentations(image, selected, seed=123)
    second, _ = apply_augmentations(image, selected, seed=123)
    assert labels == ["Brightness", "Noise", "Haze"]
    assert np.array_equal(first, second)
    assert np.any(first != image)


def test_pipeline_rejects_empty_or_unknown_augmentation(image):
    with pytest.raises(ValueError):
        apply_augmentations(image, [])
    with pytest.raises(ValueError):
        apply_augmentations(image, [{"name": "unknown"}])


def test_augmentation_rejects_invalid_parameters(image):
    with pytest.raises(ValueError):
        gamma_correction(image, 0)
    with pytest.raises(ValueError):
        add_noise(image, 1.5)


def test_brightness_increase_and_decrease_are_directional(image):
    brighter = increase_brightness(image, 0.2)
    darker = decrease_brightness(image, 0.2)
    assert brighter.mean() > image.mean()
    assert darker.mean() < image.mean()


def test_all_supported_augmentations_reproduce_with_seed(image):
    selected = [
        {"name": "brightness", "params": {"intensity": 0.2}},
        {"name": "contrast", "params": {"factor": 1.3}},
        {"name": "gamma", "params": {"gamma": 1.4}},
        {"name": "shadow", "params": {"intensity": 0.3}},
        {"name": "gaussian_blur", "params": {"strength": 0.3}},
        {"name": "motion_blur", "params": {"strength": 0.3}},
        {"name": "noise", "params": {"intensity": 0.1}},
        {"name": "occlusion", "params": {"intensity": 0.2}},
        {"name": "haze", "params": {"intensity": 0.2}},
    ]
    first, labels = apply_augmentations(image, selected, seed=2026)
    second, second_labels = apply_augmentations(image, selected, seed=2026)
    assert np.array_equal(first, second)
    assert labels == second_labels
    assert len(labels) == 9
    assert first.shape == image.shape
    assert np.any(first != image)


def test_rejects_boolean_and_nonfinite_pipeline_parameters(image):
    with pytest.raises(ValueError):
        apply_augmentations(image, [{"name": "contrast", "params": {"factor": True}}])
    with pytest.raises(ValueError):
        apply_augmentations(image, [{"name": "gamma", "params": {"gamma": float("inf")}}])


def test_metrics_for_identical_and_changed_images(image):
    identical = image_metrics(image, image)
    assert identical["width"] == 128
    assert identical["height"] == 96
    assert identical["psnr"] is None
    assert identical["ssim"] == 1.0

    changed = brightness(image, 0.3)
    metrics = image_metrics(image, changed)
    assert metrics["mean_brightness"] > float(image.mean())
    assert metrics["psnr"] is not None
    assert -1 <= metrics["ssim"] <= 1
