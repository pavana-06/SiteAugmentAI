from io import BytesIO
import json
import os
import zipfile

import numpy as np
from PIL import Image

from app import create_app


class TestConfig:
    TESTING = True
    SECRET_KEY = "test-key"
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024
    UPLOAD_FOLDER = "/tmp/siteaugment-test-uploads"
    OUTPUT_FOLDER = "/tmp/siteaugment-test-outputs"
    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
    MAX_IMAGE_PIXELS = 20_000_000
    MAX_GENERATED_IMAGES = 12


def png_bytes():
    rows, cols = np.indices((48, 64))
    pixels = np.dstack(((cols * 3) % 255, (rows * 5) % 255, (rows + cols) % 255)).astype(np.uint8)
    buffer = BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return buffer.getvalue()


def encoded_image_bytes(format_name):
    rows, cols = np.indices((48, 64))
    pixels = np.dstack(((cols * 3) % 255, (rows * 5) % 255, (rows + cols) % 255)).astype(np.uint8)
    buffer = BytesIO()
    Image.fromarray(pixels).save(buffer, format=format_name)
    return buffer.getvalue()


def client(tmp_path, monkeypatch):
    monkeypatch.setattr(TestConfig, "UPLOAD_FOLDER", str(tmp_path / "uploads"))
    monkeypatch.setattr(TestConfig, "OUTPUT_FOLDER", str(tmp_path / "outputs"))
    return create_app(TestConfig).test_client()


def test_home_renders_dashboard(tmp_path, monkeypatch):
    response = client(tmp_path, monkeypatch).get("/")
    assert response.status_code == 200
    assert b"SiteAugment AI" in response.data


def test_augment_and_downloads(tmp_path, monkeypatch):
    browser = client(tmp_path, monkeypatch)
    config = {
        "augmentations": [{"name": "brightness", "params": {"intensity": 0.2}}],
        "count": 2,
        "seed": 12,
    }
    response = browser.post(
        "/augment",
        data={"image": (BytesIO(png_bytes()), "site.png"), "config": json.dumps(config)},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert len(payload["results"]) == 2
    assert payload["results"][0]["metrics"]["width"] == 64

    image_response = browser.get(payload["results"][0]["url"])
    assert image_response.status_code == 200
    assert image_response.mimetype == "image/png"

    zip_response = browser.get(f"/download-zip/{payload['session_id']}")
    assert zip_response.status_code == 200
    with zipfile.ZipFile(BytesIO(zip_response.data)) as archive:
        assert len(archive.namelist()) == 2


def test_rejects_invalid_image_and_configuration(tmp_path, monkeypatch):
    browser = client(tmp_path, monkeypatch)
    invalid_image = browser.post(
        "/augment",
        data={"image": (BytesIO(b"not an image"), "fake.png"), "config": "{}"},
        content_type="multipart/form-data",
    )
    assert invalid_image.status_code == 400

    missing_selection = browser.post(
        "/augment",
        data={"image": (BytesIO(png_bytes()), "site.png"), "config": json.dumps({"augmentations": []})},
        content_type="multipart/form-data",
    )
    assert missing_selection.status_code == 400


def test_download_routes_reject_unknown_paths(tmp_path, monkeypatch):
    browser = client(tmp_path, monkeypatch)
    assert browser.get("/download/../../etc/passwd").status_code == 404
    assert browser.get("/download-zip/not-a-session").status_code == 404


def test_jpg_jpeg_and_png_upload_formats(tmp_path, monkeypatch):
    browser = client(tmp_path, monkeypatch)
    config = {"augmentations": [{"name": "brightness", "params": {"intensity": 0.1}}], "count": 1}
    for filename, image_bytes in (
        ("site.jpg", encoded_image_bytes("JPEG")),
        ("site.jpeg", encoded_image_bytes("JPEG")),
        ("site.png", encoded_image_bytes("PNG")),
    ):
        response = browser.post(
            "/augment",
            data={"image": (BytesIO(image_bytes), filename), "config": json.dumps(config)},
            content_type="multipart/form-data",
        )
        assert response.status_code == 200, (filename, response.get_json())


def test_invalid_extension_fake_image_and_oversized_request(tmp_path, monkeypatch):
    browser = client(tmp_path, monkeypatch)
    config = {"augmentations": [{"name": "haze", "params": {"intensity": 0.2}}], "count": 1}
    fake = browser.post(
        "/augment",
        data={"image": (BytesIO(b"fake"), "site.png"), "config": json.dumps(config)},
        content_type="multipart/form-data",
    )
    assert fake.status_code == 400

    wrong_extension = browser.post(
        "/augment",
        data={"image": (BytesIO(png_bytes()), "site.gif"), "config": json.dumps(config)},
        content_type="multipart/form-data",
    )
    assert wrong_extension.status_code == 400

    oversized = browser.post(
        "/augment",
        data={"image": (BytesIO(b"x" * (10 * 1024 * 1024 + 1)), "site.png"), "config": json.dumps(config)},
        content_type="multipart/form-data",
    )
    assert oversized.status_code == 413


def test_pixel_dimension_limit(tmp_path, monkeypatch):
    browser_app = create_app(TestConfig)
    browser_app.config["MAX_IMAGE_PIXELS"] = 100
    browser_app.config["TESTING"] = True
    browser_app.config["UPLOAD_FOLDER"] = str(tmp_path / "uploads")
    browser_app.config["OUTPUT_FOLDER"] = str(tmp_path / "outputs")
    browser_app.extensions["siteaugment_sessions"] = {}
    browser_app.extensions["siteaugment_files"] = set()
    import os
    os.makedirs(browser_app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(browser_app.config["OUTPUT_FOLDER"], exist_ok=True)
    browser = browser_app.test_client()
    response = browser.post(
        "/augment",
        data={
            "image": (BytesIO(png_bytes()), "site.png"),
            "config": json.dumps({"augmentations": [{"name": "haze"}], "count": 1}),
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert b"dimensions" in response.data


def test_seeded_generation_is_reproducible(tmp_path, monkeypatch):
    browser = client(tmp_path, monkeypatch)
    config = {
        "augmentations": [
            {"name": "brightness", "params": {"intensity": 0.2}},
            {"name": "shadow", "params": {"intensity": 0.4}},
            {"name": "noise", "params": {"intensity": 0.1}},
        ],
        "count": 1,
        "seed": 9876,
    }
    first = browser.post(
        "/augment",
        data={"image": (BytesIO(png_bytes()), "site.png"), "config": json.dumps(config)},
        content_type="multipart/form-data",
    )
    second = browser.post(
        "/augment",
        data={"image": (BytesIO(png_bytes()), "site.png"), "config": json.dumps(config)},
        content_type="multipart/form-data",
    )
    assert first.status_code == second.status_code == 200
    first_payload = first.get_json()
    second_payload = second.get_json()
    first_download = browser.get(first_payload["results"][0]["url"])
    second_download = browser.get(second_payload["results"][0]["url"])
    assert first_download.data == second_download.data


def test_zip_contains_exact_generated_files(tmp_path, monkeypatch):
    browser = client(tmp_path, monkeypatch)
    response = browser.post(
        "/augment",
        data={
            "image": (BytesIO(png_bytes()), "site.png"),
            "config": json.dumps({
                "augmentations": [{"name": "gaussian_blur", "params": {"strength": 0.4}}],
                "count": 3,
                "seed": 44,
            }),
        },
        content_type="multipart/form-data",
    )
    payload = response.get_json()
    zip_response = browser.get(f"/download-zip/{payload['session_id']}")
    with zipfile.ZipFile(BytesIO(zip_response.data)) as archive:
        names = archive.namelist()
        assert len(names) == 3
        assert set(names) == {result["filename"] for result in payload["results"]}
        assert all(name.endswith(".png") for name in names)


def test_missing_output_file_and_invalid_numeric_parameters(tmp_path, monkeypatch):
    browser = client(tmp_path, monkeypatch)
    response = browser.post(
        "/augment",
        data={
            "image": (BytesIO(png_bytes()), "site.png"),
            "config": json.dumps({
                "augmentations": [{"name": "contrast", "params": {"factor": True}}],
                "count": 1,
            }),
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 400

    valid = browser.post(
        "/augment",
        data={
            "image": (BytesIO(png_bytes()), "site.png"),
            "config": json.dumps({"augmentations": [{"name": "haze"}], "count": 1}),
        },
        content_type="multipart/form-data",
    )
    payload = valid.get_json()
    filename = payload["results"][0]["filename"]
    output_path = browser.application.config["OUTPUT_FOLDER"] + "/" + filename
    os.remove(output_path)
    assert browser.get(payload["results"][0]["url"]).status_code == 404
