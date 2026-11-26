# SiteAugment AI — Construction Safety Image Augmenter

SiteAugment AI is a local web application for producing varied construction-site imagery for computer-vision dataset development. It applies real pixel-level transformations such as lighting changes, shadows, blur, noise, occlusion, and atmospheric haze. It does **not** detect PPE or measure PPE model performance.

## Problem and objectives

Images collected at construction sites vary with weather, lighting, camera movement, dust, and obstructions. This project helps a student team generate reproducible image variations to broaden a training dataset and inspect how those transformations change image appearance.

## Features

- Responsive dark dashboard with drag-and-drop JPG/PNG upload and image preview.
- Nine selectable effects, with per-effect intensity controls and combined processing.
- Generate 1–12 distinct variations with optional seeded reproducibility through the API.
- Original/augmented wipe comparison and individual PNG or ZIP downloads.
- Image dimensions, mean pixel brightness, PSNR, and a global SSIM estimate for each result.
- Upload limits, image decoding and dimension validation, generated opaque filenames, and session-scoped downloads.

## Technology

- Python 3, Flask, OpenCV, NumPy, Pillow
- HTML, CSS, vanilla JavaScript
- Pytest

## Architecture

The Flask routes in `app.py` validate requests, call the augmentation pipeline, and expose downloads. Individual transformations live in focused modules under `augmentation/`; shared image decode, PNG encoding, and metric calculation are in `utils/image_utils.py`. The browser sends the source image and a JSON recipe to `POST /augment`. Generated images are stored under opaque UUID names; session-to-output associations are held in memory and ZIP files are assembled from those known outputs.

### Project structure

```text
SiteAugmentAI/
├── app.py
├── config.py
├── requirements.txt
├── README.md
├── augmentation/
│   ├── lighting.py
│   ├── contrast.py
│   ├── shadows.py
│   ├── blur.py
│   ├── noise.py
│   ├── occlusion.py
│   ├── haze.py
│   └── pipeline.py
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

## Tests

```bash
pytest -q
```

Tests cover algorithm output/validation, deterministic combined processing, the dashboard, valid and invalid uploads, generation, individual downloads, ZIP downloads, and rejected paths.

## Augmentation techniques

Brightness varies randomly above or below the source intensity; gamma remaps tonal response; contrast scales pixel values. Artificial shadows use a softened polygon mask. Gaussian and horizontal motion blur simulate focus and movement. Additive Gaussian noise simulates sensor grain. Occlusion places a bounded translucent rectangle over part of the image. Haze combines a smooth spatial veil with atmospheric light. Effects are applied in recipe order and retain the original width and height.

## Metrics and limitations

Mean brightness describes output pixel intensity. PSNR and SSIM compare each output to its source and are image-change/quality indicators only—not PPE detection accuracy, safety compliance, or model performance. SSIM here is a lightweight global luminance/contrast/structure estimate rather than a windowed benchmark implementation. Augmentation is synthetic and cannot represent every real site condition; strong settings can create unrealistic images. The current app keeps generated files and session associations locally, with in-memory session metadata; restarting the process invalidates ZIP sessions. Use appropriate data consent and review outputs before training.

## Future enhancements

- Persistent session metadata, automatic output expiration, and managed storage.
- More natural semantic occlusions, weather and camera artifacts, and batch dataset workflows.
- Optional perceptual metrics, background jobs, user authentication, and deployment hardening.
