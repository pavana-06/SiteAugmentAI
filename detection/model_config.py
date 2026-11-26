"""Pinned public model metadata for the optional PPE inference module."""

MODEL_REPO_ID = "Tanishjain9/yolov8n-ppe-detection-6classes"
MODEL_REVISION = "a2c1a97646ade96e48adda74781dd455d0456c40"
MODEL_FILENAME = "best.pt"
DEFAULT_CONFIDENCE = 0.25
MIN_CONFIDENCE = 0.01
MAX_CONFIDENCE = 0.99

# These are the six labels declared by the selected model card. Runtime labels
# are read from the loaded checkpoint and returned with each response.
MODEL_CARD_CLASSES = (
    "Gloves",
    "Vest",
    "goggles",
    "helmet",
    "mask",
    "safety_shoe",
)
