"""
detector.py
-----------
Loads a pretrained YOLO model (Ultralytics) and runs object detection
on a single image / video frame.

We do NOT train anything here. The model was already trained by
Ultralytics on the COCO dataset (80 everyday object classes).
"""

from pathlib import Path

from ultralytics import YOLO

# Folder where the model weights are kept (created automatically).
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Lightweight "nano" model - small and fast enough for a normal laptop CPU.
DEFAULT_MODEL = "yolo11n.pt"


class ObjectDetector:
    """Small wrapper around a pretrained Ultralytics YOLO model."""

    def __init__(self, model_name=DEFAULT_MODEL):
        # If the weights file is not in models/ yet, Ultralytics downloads
        # it automatically from its official GitHub releases (first run only).
        model_path = MODELS_DIR / model_name
        self.model = YOLO(str(model_path))

        # Dictionary of class id -> class name, e.g. {0: 'person', 39: 'bottle', ...}
        self.class_names = self.model.names

    def detect(self, frame, confidence=0.5):
        """
        Run YOLO on one BGR frame (as read by OpenCV).

        Returns a list of detections. Each detection is a dictionary:
            {
                "box": (x1, y1, x2, y2),   # pixel coordinates
                "class_id": 0,
                "label": "person",
                "confidence": 0.95
            }
        Only detections with score >= confidence are returned.
        """
        # verbose=False stops Ultralytics from printing a line for every frame
        results = self.model(frame, conf=confidence, verbose=False)
        result = results[0]  # we passed one frame, so there is one result

        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            class_id = int(box.cls[0])
            score = float(box.conf[0])

            detections.append(
                {
                    "box": (int(x1), int(y1), int(x2), int(y2)),
                    "class_id": class_id,
                    "label": self.class_names[class_id],
                    "confidence": score,
                }
            )

        return detections
