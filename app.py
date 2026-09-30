"""
Real-Time Object Detection using YOLO
-------------------------------------
Pipeline:
    Webcam -> Streamlit UI -> OpenCV frame capture -> pretrained YOLO
    -> bounding boxes + labels + confidence -> Streamlit display

Run with:
    streamlit run app.py
"""

import time

import cv2
import numpy as np
import streamlit as st

from src.detector import DEFAULT_MODEL, ObjectDetector
from src.utils import calculate_fps, count_objects, draw_detections

WEBCAM_ERROR = (
    "Unable to access webcam. "
    "Please check your camera permissions or connected webcam."
)


@st.cache_resource
def load_detector(model_name):
    """Load the YOLO model only once (Streamlit keeps it in memory)."""
    return ObjectDetector(model_name)


def show_stats(stats_box, fps, detections):
    """Show FPS, total object count and per-class counts next to the video."""
    counts = count_objects(detections)

    text = ""
    if fps is not None:
        text += f"**FPS:** {fps:.1f}\n\n"
    text += f"**Objects detected:** {len(detections)}\n\n"

    if counts:
        for label, number in sorted(counts.items()):
            text += f"- {label}: {number}\n"
    else:
        text += "_No objects above the threshold._"

    stats_box.markdown(text)


def run_webcam(detector, camera_index, confidence, frame_box, stats_box):
    """Read frames from the webcam, run YOLO on each one and display it."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        frame_box.error(WEBCAM_ERROR)
        return

    # 640x480 is enough for YOLO and keeps things fast on a normal laptop
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    previous_time = time.time()
    try:
        # Loop until the user switches detection off. Streamlit then stops
        # this script run and the 'finally' block releases the camera.
        while True:
            success, frame = cap.read()
            if not success:
                frame_box.error(WEBCAM_ERROR)
                break

            detections = detector.detect(frame, confidence)
            output = draw_detections(frame, detections)
            fps, previous_time = calculate_fps(previous_time)

            # OpenCV uses BGR colour order, Streamlit expects RGB
            output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            frame_box.image(output_rgb, width="stretch")
            show_stats(stats_box, fps, detections)
    finally:
        cap.release()


def run_image(detector, uploaded_file, confidence, frame_box, stats_box):
    """Run YOLO once on an uploaded image."""
    file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        frame_box.error("Could not read this image file.")
        return

    detections = detector.detect(image, confidence)
    output = draw_detections(image, detections)

    frame_box.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), width="stretch")
    show_stats(stats_box, None, detections)


# ---------------------------------------------------------------- page ----
st.set_page_config(page_title="Real-Time Object Detection", layout="wide")

st.title("Real-Time Object Detection using YOLO")
st.write(
    "A simple real-time object detection system using YOLO, OpenCV and Streamlit."
)

with st.spinner("Loading YOLO model..."):
    detector = load_detector(DEFAULT_MODEL)

# ------------------------------------------------------------ controls ----
st.sidebar.header("Settings")
st.sidebar.caption(f"Model: {DEFAULT_MODEL} (pretrained on COCO, 80 classes)")
source = st.sidebar.radio("Input source", ["Webcam", "Image"])
confidence = st.sidebar.slider(
    "Confidence threshold",
    min_value=0.10,
    max_value=1.00,
    value=0.50,
    step=0.05,
    help="Detections with a lower confidence score are hidden.",
)

# -------------------------------------------------------------- output ----
video_col, stats_col = st.columns([3, 1])
frame_box = video_col.empty()
stats_col.subheader("Detections")
stats_box = stats_col.empty()

if source == "Webcam":
    camera_index = st.sidebar.number_input(
        "Camera index",
        min_value=0,
        max_value=5,
        value=0,
        help="0 is usually the built-in laptop webcam.",
    )
    run = st.sidebar.toggle("Start detection")

    if run:
        run_webcam(detector, int(camera_index), confidence, frame_box, stats_box)
    else:
        frame_box.info("Switch on **Start detection** in the sidebar to open the webcam.")

else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        run_image(detector, uploaded_file, confidence, frame_box, stats_box)
    else:
        frame_box.info("Upload an image from the sidebar to run detection on it.")
