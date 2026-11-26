# SiteAugment AI — Construction Safety Image Augmenter

SiteAugment AI is a local web application for producing varied construction-site imagery and running optional pretrained PPE object detection for computer-vision research. Augmentation applies real pixel-level transformations such as lighting changes, shadows, blur, noise, occlusion, and atmospheric haze. PPE detections are model outputs, not safety compliance decisions or evidence of improved model performance.

## Problem and objectives

Images collected at construction sites vary with weather, lighting, camera movement, dust, and obstructions. This project helps a student team generate reproducible image variations to broaden a training dataset and inspect how those transformations change image appearance.

## Features

- Responsive dark dashboard with drag-and-drop JPG/PNG upload and image preview.
- Nine selectable effects, with per-effect intensity controls and combined processing.
- Generate 1–12 distinct variations with optional seeded reproducibility through the API.
- Original/augmented wipe comparison and individual PNG or ZIP downloads.
- Image dimensions, mean pixel brightness, PSNR, and a global SSIM estimate for each result.
- On-demand PPE detections for original and generated images, including boxes, scores, class counts, and separately downloadable annotated PNGs.
- Upload limits, image decoding and dimension validation, generated opaque filenames, and session-scoped downloads.

## Technology

- Python 3, Flask, OpenCV, NumPy, Pillow
- HTML, CSS, vanilla JavaScript
- Pytest

## Architecture

The Flask routes in `app.py` validate requests, call the augmentation and detection modules, and expose downloads. Individual transformations live in focused modules under `augmentation/`; the lazy-loaded ONNX Runtime detector, inference and visualization are in `detection/`; shared image decode, PNG encoding, and augmentation metrics are in `utils/image_utils.py`. The browser sends the source image and a JSON recipe to `POST /augment`; `POST /detect` runs the model on the original upload and `POST /detect-augmented` accepts only generated filenames belonging to a known in-memory augmentation session. The ONNX model loads at most once per app process and only when detection is first requested. Generated and annotated images use opaque UUID filenames; only registered outputs can be previewed or downloaded.

## PPE detection model

The detector uses the public [Tanishjain9 YOLOv8n PPE Detection (6 Classes) checkpoint](https://huggingface.co/Tanishjain9/yolov8n-ppe-detection-6classes), pinned to revision `a2c1a97646ade96e48adda74781dd455d0456c40`. Its model card reports these six output labels:

| Class label | Meaning in the model card |
| --- | --- |
| `Gloves` | Gloves |
| `Vest` | Safety/reflective vest |
| `goggles` | Eye protection |
| `helmet` | Safety helmet |
| `mask` | Face mask |
| `safety_shoe` | Safety footwear |

This checkpoint does **not** advertise a `person` class, `no-helmet`/`no-vest` classes, or a PPE compliance classification. The model card declares the checkpoint license as **MIT**. Its card describes a custom Roboflow-format PPE dataset but does not identify a specific dataset/version or establish the dataset's redistribution license; verify upstream data rights before redistributing derived weights or datasets. The repository bundles a 10.5 MiB ONNX export of this pinned checkpoint at `models/ppe-yolov8n.onnx`. It runs with CPU ONNX Runtime and does not require PyTorch or Ultralytics in the Vercel function. The ONNX export preserves the checkpoint's six classes listed above.

The ONNX export was generated from the pinned upstream checkpoint using [Ultralytics](https://github.com/ultralytics/ultralytics). Ultralytics is licensed under AGPL-3.0 (with an Enterprise license option); review the [official license guidance](https://www.ultralytics.com/license) before redistributing the export. For optional local `.pt` checkpoint loading or model export, install `requirements-detection.txt`. If you configure a custom `PPE_MODEL_PATH`, that file's class names and license become your responsibility.

### Project structure

```text
SiteAugmentAI/
├── app.py
├── config.py
├── requirements.txt
├── vercel.json
├── README.md
├── models/ppe-yolov8n.onnx
├── augmentation/
│   ├── lighting.py
│   ├── contrast.py
│   ├── shadows.py
│   ├── blur.py
│   ├── noise.py
│   ├── occlusion.py
│   ├── haze.py
│   └── pipeline.py
├── detection/
│   ├── detector.py
│   └── model_config.py
├── utils/image_utils.py
├── templates/index.html
├── static/css/style.css
├── static/js/main.js
├── uploads/
├── outputs/
└── tests/
```

## Setup and run

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000). For deployment, set a strong `SECRET_KEY`, disable Flask debug mode, and use a production WSGI server.

PPE inference uses the bundled ONNX model and the lightweight CPU runtime included in `requirements.txt`; no model download is needed at runtime.

To use a custom Ultralytics `.pt` checkpoint instead, set `PPE_MODEL_PATH=/absolute/path/to/model.pt` before starting Flask and install the optional dependencies:

```bash
python -m pip install -r requirements-detection.txt
```

Ultralytics may install the GUI OpenCV wheel. On a headless Linux machine without `libGL.so.1`, if OpenCV imports fail after installing those optional dependencies, restore the headless wheel:

```bash
python -m pip uninstall -y opencv-python
python -m pip install --force-reinstall --no-deps opencv-python-headless
```

Because the headless OpenCV wheel provides the same `cv2` runtime as Ultralytics' GUI wheel, `pip check` may still report that Ultralytics' `opencv-python` package requirement is not installed after applying this headless-server workaround. OpenCV inference and this application work with the installed headless replacement; do not install both OpenCV wheels into a server environment lacking `libGL.so.1`.

### Deploy on Vercel

The app uses CPU ONNX Runtime and a bundled 10.5 MiB model instead of PyTorch, avoiding the multi-gigabyte inference dependency bundle. `vercel.json` explicitly includes the model in the Flask function package. Import the GitHub repository into Vercel, keep the Python build using `requirements.txt`, and set `SECRET_KEY` in the project environment variables. If `PPE_MODEL_PATH` was previously set to a `.pt` file, remove it so the bundled ONNX model is used. Redeploy after pushing changes, then test **Detect PPE in original**. The first request may take longer while the serverless function initializes the ONNX session.

Vercel functions have ephemeral storage and separate instances; in-memory session associations are not shared across instances. Original-image detection works independently, but augmented-image detection and downloads may not persist reliably across separate function invocations. For durable multi-request workflows, use shared object storage and a persistent session store or deploy on a long-running web service.

The Render Blueprint in `render.yaml` also uses the bundled ONNX Runtime model, so it does not install PyTorch. Free instances may sleep when idle and have ephemeral local storage; if inference exceeds the selected instance's memory, use a larger plan.

## PPE detection usage and API

1. Upload a supported JPG/JPEG/PNG.
2. Select **Detect PPE in original** and choose a confidence threshold, or generate image variations first.
3. Use **Detect PPE** on any generated image to run inference on that augmented output.
4. Review the class labels, confidence scores, counts, and annotated image; download the overlay separately from the untouched source/output image.

API endpoints:

- `POST /detect`: multipart form fields `image` and optional numeric `confidence`.
- `POST /detect-augmented`: JSON fields `session_id`, `filename`, and optional `confidence`; the filename must belong to that generated session.
- `GET /preview/<filename>`: inline preview for a registered generated/annotated PNG.
- `GET /download/<filename>`: download a registered PNG.

The response includes pixel-coordinate boxes (`x1`, `y1`, `x2`, `y2`), class id and label, confidence from 0 to 1, all class counts (including zero counts), confidence min/mean/max, and an annotated-preview URL. The overlay is drawn on a copy; original image pixels are not modified.

## Tests

```bash
pytest -q
```

Tests cover algorithm output/validation, deterministic combined processing, the dashboard, valid and invalid uploads, generation, individual downloads, ZIP downloads, rejected paths, mocked PPE inference, ONNX preprocessing/postprocessing, visualization, and endpoints. The bundled ONNX model also supports a real inference smoke test without downloading weights.

## Augmentation techniques

Brightness varies randomly above or below the source intensity; gamma remaps tonal response; contrast scales pixel values. Artificial shadows use a softened polygon mask. Gaussian and horizontal motion blur simulate focus and movement. Additive Gaussian noise simulates sensor grain. Occlusion places a bounded translucent rectangle over part of the image. Haze combines a smooth spatial veil with atmospheric light. Effects are applied in recipe order and retain the original width and height.

## Metrics and limitations

Mean brightness describes output pixel intensity. PSNR and SSIM compare each output to its source and are image-change/quality indicators only—not PPE detection accuracy, safety compliance, or model performance. Detection counts and confidence distributions are descriptive inference outputs only. Comparing detection counts before and after augmentation does not establish improved model accuracy; that requires a controlled, properly designed model-training and evaluation experiment with held-out data and suitable metrics. The selected model may miss or misclassify objects, especially under low light, blur, occlusion, small object sizes, unseen PPE styles, or conditions outside its training domain. Always require human review; do not use these results as the sole basis for safety or employment decisions. SSIM here is a lightweight global luminance/contrast/structure estimate rather than a windowed benchmark implementation. Augmentation is synthetic and cannot represent every real site condition; strong settings can create unrealistic images. The current app keeps generated files and session associations locally, with in-memory session metadata; restarting the process invalidates ZIP sessions. Use appropriate data consent and review outputs before training.

## Future enhancements

- Persistent session metadata, automatic output expiration, and managed storage.
- More natural semantic occlusions, weather and camera artifacts, and batch dataset workflows.
- Optional perceptual metrics, background jobs, user authentication, and deployment hardening.
