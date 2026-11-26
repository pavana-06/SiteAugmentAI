from io import BytesIO
import json
import os
import re
import uuid
import zipfile

from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from augmentation import SUPPORTED_AUGMENTATIONS, apply_augmentations
from config import Config
from utils.image_utils import decode_image, encode_png, image_metrics


def create_app(config_object=Config):
    app = Flask(__name__)
    app.config.from_object(config_object)
    app.config["UPLOAD_FOLDER"] = str(app.config["UPLOAD_FOLDER"])
    app.config["OUTPUT_FOLDER"] = str(app.config["OUTPUT_FOLDER"])
    app.extensions["siteaugment_sessions"] = {}
    app.extensions["siteaugment_files"] = set()

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)

    @app.errorhandler(RequestEntityTooLarge)
    def handle_large_upload(_error):
        return jsonify(error="Upload is too large. Maximum file size is 10 MB."), 413

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/augment")
    def augment():
        upload = request.files.get("image")
        if upload is None or not upload.filename:
            return jsonify(error="Choose an image to upload."), 400
        safe_upload_name = secure_filename(upload.filename)
        extension = safe_upload_name.rsplit(".", 1)[-1].lower() if "." in safe_upload_name else ""
        if extension not in app.config["ALLOWED_EXTENSIONS"]:
            return jsonify(error="Only JPG, JPEG, and PNG images are supported."), 400
        try:
            original = decode_image(
                upload.read(),
                app.config["ALLOWED_EXTENSIONS"],
                app.config["MAX_IMAGE_PIXELS"],
            )
        except ValueError as exc:
            return jsonify(error=str(exc)), 400

        try:
            settings = json.loads(request.form.get("config", "{}"))
        except (TypeError, json.JSONDecodeError):
            return jsonify(error="Configuration must be valid JSON."), 400
        if not isinstance(settings, dict):
            return jsonify(error="Configuration must be a JSON object."), 400
        selected = settings.get("augmentations")
        count = settings.get("count", 4)
        seed = settings.get("seed")
        if not isinstance(selected, list) or not selected:
            return jsonify(error="Select at least one augmentation."), 400
        if len(selected) > len(SUPPORTED_AUGMENTATIONS):
            return jsonify(error="Too many augmentations selected."), 400
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= app.config["MAX_GENERATED_IMAGES"]:
            return jsonify(error=f"Image count must be between 1 and {app.config['MAX_GENERATED_IMAGES']}."), 400
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1):
            return jsonify(error="Seed must be an integer between 0 and 4294967295."), 400
        if seed is not None and seed + count - 1 > 2**32 - 1:
            return jsonify(error="The seed and generated-image count exceed the supported seed range."), 400
        for item in selected:
            if not isinstance(item, dict) or item.get("name") not in SUPPORTED_AUGMENTATIONS:
                return jsonify(error="One or more selected augmentations are not supported."), 400

        session_id = uuid.uuid4().hex
        results = []
        output_names = []
        for index in range(count):
            try:
                augmented, labels = apply_augmentations(
                    original, selected, None if seed is None else seed + index
                )
            except (ValueError, TypeError, OverflowError) as exc:
                return jsonify(error=f"Invalid augmentation settings: {exc}"), 400
            filename = f"{uuid.uuid4().hex}.png"
            output_path = os.path.join(app.config["OUTPUT_FOLDER"], filename)
            with open(output_path, "wb") as output_file:
                output_file.write(encode_png(augmented))
            output_names.append(filename)
            results.append({
                "filename": filename,
                "url": f"/download/{filename}",
                "labels": labels,
                "metrics": image_metrics(original, augmented),
            })

        app.extensions["siteaugment_sessions"][session_id] = tuple(output_names)
        app.extensions["siteaugment_files"].update(output_names)
        return jsonify(
            session_id=session_id,
            original={"width": int(original.shape[1]), "height": int(original.shape[0])},
            results=results,
        )

    @app.get("/download/<filename>")
    def download(filename):
        if not re.fullmatch(r"[a-f0-9]{32}\.png", filename):
            return jsonify(error="Image not found."), 404
        if filename not in app.extensions["siteaugment_files"]:
            return jsonify(error="Image not found."), 404
        output_path = os.path.join(app.config["OUTPUT_FOLDER"], filename)
        if not os.path.isfile(output_path):
            return jsonify(error="Image not found."), 404
        return send_file(
            output_path,
            mimetype="image/png",
            as_attachment=True,
            download_name=filename,
        )

    @app.get("/download-zip/<session_id>")
    def download_zip(session_id):
        if not re.fullmatch(r"[a-f0-9]{32}", session_id):
            return jsonify(error="Session not found."), 404
        names = app.extensions["siteaugment_sessions"].get(session_id)
        if not names:
            return jsonify(error="Session not found."), 404
        archive = BytesIO()
        added_files = 0
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zipped:
            for filename in names:
                if filename in app.extensions["siteaugment_files"]:
                    output_path = os.path.join(app.config["OUTPUT_FOLDER"], filename)
                    if os.path.isfile(output_path):
                        zipped.write(output_path, arcname=filename)
                        added_files += 1
        if added_files == 0:
            return jsonify(error="Session not found."), 404
        archive.seek(0)
        return send_file(
            archive,
            mimetype="application/zip",
            as_attachment=True,
            download_name=f"siteaugment-{session_id[:8]}.zip",
        )

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG", "0") == "1")
