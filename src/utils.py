"""
utils.py
--------
Small helper functions used by the app:
    - drawing bounding boxes and labels with OpenCV
    - formatting the label text
    - calculating FPS
    - counting objects in a frame
"""

import time
from collections import Counter

import cv2

# A few fixed BGR colours. Each class always gets the same colour
# (class_id % number of colours), so e.g. "person" is always the same colour.
COLORS = [
    (0, 200, 0),      # green
    (255, 128, 0),    # blue
    (0, 128, 255),    # orange
    (200, 0, 200),    # purple
    (0, 220, 220),    # yellow
    (255, 255, 0),    # cyan
    (60, 60, 230),    # red
    (180, 180, 180),  # grey
]


def format_label(label, confidence):
    """Return text like 'person 0.95'."""
    return f"{label} {confidence:.2f}"


def draw_detections(frame, detections):
    """
    Draw a rectangle and a 'name confidence' label for every detection.
    Works on a copy, so the original frame is not changed.
    """
    output = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        color = COLORS[det["class_id"] % len(COLORS)]
        text = format_label(det["label"], det["confidence"])

        # 1. Bounding box
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

        # 2. Filled background behind the text so it is easy to read
        (text_w, text_h), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        label_y = max(y1, text_h + baseline + 4)  # keep label inside the image
        cv2.rectangle(
            output,
            (x1, label_y - text_h - baseline - 4),
            (x1 + text_w + 4, label_y),
            color,
            -1,  # -1 means filled
        )

        # 3. Label text (white)
        cv2.putText(
            output,
            text,
            (x1 + 2, label_y - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    return output


def calculate_fps(previous_time):
    """
    FPS = 1 / (time taken for one loop).
    Returns (fps, current_time) so the caller can pass current_time
    back in on the next frame.
    """
    current_time = time.time()
    elapsed = current_time - previous_time
    fps = 1.0 / elapsed if elapsed > 0 else 0.0
    return fps, current_time


def count_objects(detections):
    """
    Count how many objects of each class are in ONE frame,
    e.g. {'person': 2, 'bottle': 1}.
    This is a simple per-frame count, not object tracking.
    """
    return dict(Counter(det["label"] for det in detections))
