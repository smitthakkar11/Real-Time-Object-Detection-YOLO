"""
Real-Time Object Detection using YOLO
-------------------------------------
Pipeline:
    Webcam -> Streamlit UI -> OpenCV frame capture -> pretrained YOLO
    -> bounding boxes + labels + confidence -> Streamlit display

Input sources:
    - Browser webcam (real-time): the browser sends the video to the app with
      WebRTC (streamlit-webrtc). Works locally and when the app is deployed online.
    - Local webcam (OpenCV): the app opens the camera itself with cv2.VideoCapture.
      Only works when the app runs on the same computer as the webcam.
    - Image: run detection on one uploaded photo.

Run with:
    streamlit run app.py
"""

import threading
import time

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from src.detector import DEFAULT_MODEL, ObjectDetector
from src.utils import calculate_fps, count_objects, draw_detections

WEBCAM_ERROR = (
    "Unable to access webcam. "
    "Please check your camera permissions or connected webcam."
)
ONLINE_HINT = (
    " If you are using the online version of this app, choose "
    "**Browser webcam (real-time)** instead: the server has no webcam of its own."
)

BROWSER_WEBCAM = "Browser webcam (real-time)"
LOCAL_WEBCAM = "Local webcam (OpenCV)"
IMAGE = "Image"


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
        frame_box.error(WEBCAM_ERROR + ONLINE_HINT)
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


def run_browser_webcam(detector, confidence, stats_box):
    """
    Real-time detection on video sent by the browser (works locally and online).

    streamlit-webrtc calls process_frame() for every video frame in a separate
    thread. Streamlit functions cannot be used there, so the latest FPS and
    detections are stored in a small dictionary protected by a lock, and the
    page reads them a few times per second.
    """
    if "live" not in st.session_state:
        st.session_state.live = {
            "lock": threading.Lock(),
            "previous_time": time.time(),
            "fps": 0.0,
            "detections": [],
        }
    live = st.session_state.live

    def process_frame(frame):
        image = frame.to_ndarray(format="bgr24")  # WebRTC frame -> OpenCV BGR array
        detections = detector.detect(image, confidence)
        output = draw_detections(image, detections)
        with live["lock"]:
            live["fps"], live["previous_time"] = calculate_fps(live["previous_time"])
            live["detections"] = detections
        return av.VideoFrame.from_ndarray(output, format="bgr24")

    ctx = webrtc_streamer(
        key="browser-webcam",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=process_frame,
        media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
        async_processing=True,  # drop frames that arrive while YOLO is busy
    )

    if not ctx.state.playing:
        stats_box.info("Click **START** and allow camera access in your browser.")
    while ctx.state.playing:
        with live["lock"]:
            fps, detections = live["fps"], live["detections"]
        show_stats(stats_box, fps, detections)
        time.sleep(0.5)


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
source = st.sidebar.radio(
    "Input source",
    [BROWSER_WEBCAM, LOCAL_WEBCAM, IMAGE],
    help=(
        "Browser webcam works on your computer and online. "
        "Local webcam only works when the app runs on the computer with the webcam."
    ),
)
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

if source == BROWSER_WEBCAM:
    with video_col:
        run_browser_webcam(detector, confidence, stats_box)

elif source == LOCAL_WEBCAM:
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
