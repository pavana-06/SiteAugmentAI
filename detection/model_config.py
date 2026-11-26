"""Pinned public model metadata for the optional PPE inference module."""

MODEL_REPO_ID = "Tanishjain9/yolov8n-ppe-detection-6classes"
MODEL_REVISION = "a2c1a97646ade96e48adda74781dd455d0456c40"
MODEL_FILENAME = "best.pt"
ONNX_MODEL_FILENAME = "ppe-yolov8n.onnx"
ONNX_INPUT_SIZE = 640
ONNX_NMS_IOU_THRESHOLD = 0.7
ONNX_MAX_CANDIDATES = 3000
ONNX_MAX_DETECTIONS = 300
DEFAULT_CONFIDENCE = 0.25
MIN_CONFIDENCE = 0.01
MAX_CONFIDENCE = 0.99

# These are the six labels and order declared by the pinned model checkpoint.
MODEL_CARD_CLASSES = (
    "Gloves",
    "Vest",
    "goggles",
    "helmet",
    "mask",
    "safety_shoe",
)
