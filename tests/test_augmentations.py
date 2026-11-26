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


def varied_images():
    rows, cols = np.indices((96, 128))
    gradient = np.stack(
        ((cols * 2) % 256, (rows * 2) % 256, (rows + cols) % 256),
        axis=-1,
    ).astype(np.uint8)
    grid = (((rows // 8 + cols // 8) % 2) * 170 + 40).astype(np.uint8)
    structured = np.stack((grid, np.roll(grid, 3, axis=0), np.roll(grid, 3, axis=1)), axis=-1)
    texture = np.random.default_rng(26).integers(30, 226, size=(80, 112, 3), dtype=np.uint8)
    return [gradient, structured, texture]


BALANCED_SITE_RECIPE = [
    {"name": "brightness", "params": {"intensity": 0.12}},
    {"name": "contrast", "params": {"factor": 1.1}},
    {"name": "shadow", "params": {"intensity": 0.12}},
    {"name": "noise", "params": {"intensity": 0.04}},
    {"name": "haze", "params": {"intensity": 0.08}},
]


def edge_energy(image):
    grayscale = image.astype(np.float32).mean(axis=2)
    return (
        np.abs(np.diff(grayscale, axis=0)).mean()
        + np.abs(np.diff(grayscale, axis=1)).mean()
    )


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


def test_default_effects_change_multiple_images_and_preserve_dimensions():
    images = varied_images()
    for name in (
        "brightness", "contrast", "gamma", "shadow", "gaussian_blur",
        "motion_blur", "noise", "occlusion", "haze",
    ):
        changed = False
        for source in images:
            result, _ = apply_augmentations(source, [{"name": name}], seed=73)
            assert result.shape == source.shape
            assert result.dtype == np.uint8
            changed |= bool(np.any(result != source))
        assert changed, f"{name} did not change any test image"


def test_default_blur_retains_structural_detail():
    images = varied_images()
    for name in ("gaussian_blur", "motion_blur"):
        changed = False
        for detail_image in images:
            original_energy = edge_energy(detail_image)
            result, _ = apply_augmentations(detail_image, [{"name": name}], seed=19)
            changed |= bool(np.any(result != detail_image))
            assert edge_energy(result) / original_energy >= 0.6
        assert changed


def test_default_noise_remains_subtle():
    source = np.full((64, 64, 3), 128, dtype=np.uint8)
    result, _ = apply_augmentations(source, [{"name": "noise"}], seed=19)
    noise = result.astype(np.int16) - source.astype(np.int16)
    assert 0.5 < float(noise.std()) < 4.0


def test_balanced_recipe_is_distinct_and_reproducible_across_images():
    for source in varied_images():
        first, _ = apply_augmentations(source, BALANCED_SITE_RECIPE, seed=1337)
        repeated, _ = apply_augmentations(source, BALANCED_SITE_RECIPE, seed=1337)
        assert np.array_equal(first, repeated)
        assert first.shape == source.shape
        assert first.dtype == np.uint8
        assert np.any(first != source)

        variations = [
            apply_augmentations(source, BALANCED_SITE_RECIPE, seed=seed)[0]
            for seed in (51, 52, 53, 54)
        ]
        for first_index in range(len(variations)):
            for second_index in range(first_index + 1, len(variations)):
                difference = np.mean(np.abs(
                    variations[first_index].astype(np.int16)
                    - variations[second_index].astype(np.int16)
                ))
                assert difference > 1.0


def test_pipeline_rejects_duplicate_effects(image):
    with pytest.raises(ValueError, match="only be selected once"):
        apply_augmentations(image, [{"name": "noise"}, {"name": "noise"}])


@pytest.mark.parametrize("name,parameter,value", [
    ("brightness", "intensity", 0.41),
    ("contrast", "factor", 1.51),
    ("gamma", "gamma", 1.81),
    ("shadow", "intensity", 0.61),
    ("gaussian_blur", "strength", 0.61),
    ("motion_blur", "strength", 0.61),
    ("noise", "intensity", 0.26),
    ("occlusion", "intensity", 0.26),
    ("haze", "intensity", 0.41),
])
def test_pipeline_rejects_excessive_parameters(image, name, parameter, value):
    with pytest.raises(ValueError):
        apply_augmentations(
            image,
            [{"name": name, "params": {parameter: value}}],
        )


def test_pipeline_softens_combined_blur_effects():
    source = varied_images()[1]
    recipe = [
        {"name": "gaussian_blur", "params": {"strength": 0.6}},
        {"name": "motion_blur", "params": {"strength": 0.6}},
    ]
    result, _ = apply_augmentations(source, recipe, seed=19)
    repeated, _ = apply_augmentations(source, recipe, seed=19)
    assert result.shape == source.shape
    assert np.array_equal(result, repeated)
    assert np.any(result != source)
    assert edge_energy(result) / edge_energy(source) >= 0.5
