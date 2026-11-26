from io import BytesIO
import json
import logging
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from app import create_app
from config import default_data_dir
from detection.detector import DetectorUnavailable


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


class FakePpeDetector:
    def __init__(self):
        self.calls = []

    def detect(self, image, confidence):
        self.calls.append((image.copy(), confidence))
        height, width = image.shape[:2]
        x2, y2 = min(25, width), min(20, height)
        detections = []
        if x2 > 1 and y2 > 1:
            detections.append({
                "class_id": 3,
                "label": "helmet",
                "confidence": 0.91,
                "bbox": {"x1": 1.0, "y1": 1.0, "x2": float(x2), "y2": float(y2)},
            })
        return {
            "detections": detections,
            "summary": {
                "total_detections": len(detections),
                "class_counts": {"helmet": len(detections), "vest": 0},
                "confidence": {
                    "min": 0.91 if detections else None,
                    "max": 0.91 if detections else None,
                    "mean": 0.91 if detections else None,
                },
            },
            "classes": ["helmet", "vest"],
        }


class UnavailablePpeDetector:
    def detect(self, _image, _confidence):
        raise DetectorUnavailable("PPE detection model is unavailable. Check model setup and server logs.")


def ppe_client(tmp_path, monkeypatch, detector):
    browser = client(tmp_path, monkeypatch)
    browser.application.config["PPE_DETECTOR_FACTORY"] = lambda: detector
    return browser


def test_home_renders_dashboard(tmp_path, monkeypatch):
    response = client(tmp_path, monkeypatch).get("/")
    assert response.status_code == 200
    assert b"SiteAugment AI" in response.data
    assert b'id="comparison-panel"' in response.data
    assert b'id="detection-section"' in response.data


def test_vercel_defaults_to_temporary_data_directory(monkeypatch):
    monkeypatch.delenv("SITEAUGMENT_DATA_DIR", raising=False)
    monkeypatch.setenv("VERCEL", "1")
    assert default_data_dir() == Path(tempfile.gettempdir()) / "siteaugment-ai"


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


def test_augmentation_output_write_failure_returns_json_and_logs(tmp_path, monkeypatch, caplog):
    browser = client(tmp_path, monkeypatch)
    output_folder = browser.application.config["OUTPUT_FOLDER"]
    real_open = open

    def fail_output_write(path, mode="r", *args, **kwargs):
        if str(path).startswith(output_folder) and any(flag in mode for flag in "wax+"):
            raise PermissionError("private filesystem detail")
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", fail_output_write)
    with caplog.at_level(logging.ERROR):
        response = browser.post(
            "/augment",
            data={
                "image": (BytesIO(png_bytes()), "site.png"),
                "config": json.dumps({
                    "augmentations": [{"name": "brightness", "params": {"intensity": 0.1}}],
                    "count": 1,
                }),
            },
            content_type="multipart/form-data",
        )

    assert response.status_code == 500
    assert response.is_json
    assert response.get_json() == {"error": "Could not save the generated image."}
    assert "Could not save generated augmented image" in caplog.text
    assert "private filesystem detail" in caplog.text
    assert "private filesystem detail" not in response.get_data(as_text=True)


def test_unexpected_api_error_returns_safe_json_and_logs(tmp_path, monkeypatch, caplog):
    browser = client(tmp_path, monkeypatch)
    browser.application.config["TESTING"] = False

    def fail_augmentation(*_args, **_kwargs):
        raise RuntimeError("private implementation detail")

    monkeypatch.setattr("app.apply_augmentations", fail_augmentation)
    with caplog.at_level(logging.ERROR):
        response = browser.post(
            "/augment",
            data={
                "image": (BytesIO(png_bytes()), "site.png"),
                "config": json.dumps({
                    "augmentations": [{"name": "brightness", "params": {"intensity": 0.1}}],
                    "count": 1,
                }),
            },
            content_type="multipart/form-data",
        )

    assert response.status_code == 500
    assert response.is_json
    assert "private implementation detail" not in response.get_data(as_text=True)
    assert "Unhandled request error" in caplog.text
    assert "private implementation detail" in caplog.text


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


def test_original_detection_endpoint_visualization_and_download(tmp_path, monkeypatch):
    detector = FakePpeDetector()
    browser = ppe_client(tmp_path, monkeypatch, detector)
    response = browser.post(
        "/detect",
        data={
            "image": (BytesIO(png_bytes()), "site.png"),
            "confidence": "0.55",
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["source"] == "original"
    assert payload["image"] == {"width": 64, "height": 48}
    assert payload["summary"]["total_detections"] == 1
    assert payload["detections"][0]["label"] == "helmet"
    assert detector.calls[0][1] == 0.55
    assert np.array_equal(detector.calls[0][0][:, :, 0], np.asarray(Image.open(BytesIO(png_bytes())).convert("RGB"))[:, :, 0])

    preview = browser.get(payload["annotated_url"])
    assert preview.status_code == 200
    assert preview.mimetype == "image/png"
    assert "attachment" not in preview.headers.get("Content-Disposition", "")
    with Image.open(BytesIO(preview.data)) as annotated:
        assert annotated.size == (64, 48)
    download = browser.get(payload["annotated_url"].replace("/preview/", "/download/"))
    assert download.status_code == 200
    assert "attachment" in download.headers.get("Content-Disposition", "")


def test_detection_visualization_write_failure_returns_json_and_logs(tmp_path, monkeypatch, caplog):
    browser = ppe_client(tmp_path, monkeypatch, FakePpeDetector())
    output_folder = browser.application.config["OUTPUT_FOLDER"]
    real_open = open

    def fail_output_write(path, mode="r", *args, **kwargs):
        if str(path).startswith(output_folder) and any(flag in mode for flag in "wax+"):
            raise PermissionError("private filesystem detail")
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", fail_output_write)
    with caplog.at_level(logging.ERROR):
        response = browser.post(
            "/detect",
            data={"image": (BytesIO(png_bytes()), "site.png")},
            content_type="multipart/form-data",
        )

    assert response.status_code == 500
    assert response.is_json
    assert response.get_json() == {"error": "Could not save the detection visualization."}
    assert "Could not save original-image PPE visualization" in caplog.text
    assert "private filesystem detail" in caplog.text
    assert "private filesystem detail" not in response.get_data(as_text=True)


def test_detection_rejects_bad_upload_and_confidence(tmp_path, monkeypatch):
    browser = ppe_client(tmp_path, monkeypatch, FakePpeDetector())
    invalid_image = browser.post(
        "/detect",
        data={"image": (BytesIO(b"bad"), "site.png")},
        content_type="multipart/form-data",
    )
    assert invalid_image.status_code == 400
    invalid_confidence = browser.post(
        "/detect",
        data={
            "image": (BytesIO(png_bytes()), "site.png"),
            "confidence": "1.5",
        },
        content_type="multipart/form-data",
    )
    assert invalid_confidence.status_code == 400
    invalid_extension = browser.post(
        "/detect",
        data={"image": (BytesIO(png_bytes()), "site.gif")},
        content_type="multipart/form-data",
    )
    assert invalid_extension.status_code == 400
    huge_confidence = browser.post(
        "/detect",
        data={
            "image": (BytesIO(png_bytes()), "site.png"),
            "confidence": str(10**400),
        },
        content_type="multipart/form-data",
    )
    assert huge_confidence.status_code == 400


def test_detection_enforces_image_pixel_limit(tmp_path, monkeypatch):
    detector = FakePpeDetector()
    browser = ppe_client(tmp_path, monkeypatch, detector)
    browser.application.config["MAX_IMAGE_PIXELS"] = 100
    response = browser.post(
        "/detect",
        data={"image": (BytesIO(png_bytes()), "site.png")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert b"dimensions" in response.data
    assert detector.calls == []


def test_detection_model_loading_failure_returns_safe_503(tmp_path, monkeypatch):
    browser = ppe_client(tmp_path, monkeypatch, UnavailablePpeDetector())
    response = browser.post(
        "/detect",
        data={"image": (BytesIO(png_bytes()), "site.png")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 503
    assert b"/tmp/" not in response.data


def test_augmented_detection_requires_session_membership(tmp_path, monkeypatch):
    detector = FakePpeDetector()
    browser = ppe_client(tmp_path, monkeypatch, detector)
    generated = browser.post(
        "/augment",
        data={
            "image": (BytesIO(png_bytes()), "site.png"),
            "config": json.dumps({"augmentations": [{"name": "haze"}], "count": 1, "seed": 5}),
        },
        content_type="multipart/form-data",
    ).get_json()
    target = generated["results"][0]
    response = browser.post(
        "/detect-augmented",
        json={
            "session_id": generated["session_id"],
            "filename": target["filename"],
            "confidence": 0.4,
        },
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["source"] == "augmented"
    assert payload["summary"]["total_detections"] == 1
    assert detector.calls[-1][1] == 0.4
    assert browser.get(payload["annotated_url"]).status_code == 200

    wrong_session = browser.post(
        "/detect-augmented",
        json={"session_id": "0" * 32, "filename": target["filename"]},
    )
    assert wrong_session.status_code == 404
    arbitrary_filename = browser.post(
        "/detect-augmented",
        json={"session_id": generated["session_id"], "filename": "../etc/passwd"},
    )
    assert arbitrary_filename.status_code == 400


def test_detection_endpoint_requires_valid_json_object(tmp_path, monkeypatch):
    browser = ppe_client(tmp_path, monkeypatch, FakePpeDetector())
    response = browser.post("/detect-augmented", data="[]", content_type="application/json")
    assert response.status_code == 400
