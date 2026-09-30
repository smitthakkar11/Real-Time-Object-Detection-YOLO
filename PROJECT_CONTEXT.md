# PROJECT_CONTEXT

A quick summary for another developer or AI assistant picking up this project.

## Project

- **Name:** Real-Time Object Detection using YOLO
- **Course:** Image Processing and Computer Vision (college project)
- **Team:**
  - Yogesh Dixit (20233362)
  - Sumitabh Ahirwar (20233280)
  - Smit Thakkar (20233570)
  - Vishal Prabhu (20233582)

**Live demo:** https://realtime-object-detection-yolo.streamlit.app (Streamlit Community Cloud,
deployed from `main`; every push redeploys).

## Objective

Show how a **pretrained** YOLO model can be combined with OpenCV to detect objects in
real time from a laptop webcam, with a very basic Streamlit UI. Each processed frame
shows bounding boxes, class names and confidence scores, plus FPS and a per-frame
object count. No model training is involved.

## Technologies

- Python 3.11 (tested; 3.10–3.12 should work)
- OpenCV (`opencv-python`): webcam capture, drawing, BGR→RGB conversion
- Ultralytics (`ultralytics`): YOLO model loading and inference (pulls in PyTorch)
- Streamlit: UI (the only UI framework)
- streamlit-webrtc: streams the browser webcam to the app and back (WebRTC). Added so that
  real-time detection also works in the online deployment. Pulls in aiortc + av (PyAV).
- NumPy: decoding uploaded image bytes

Deliberately **not** used: FastAPI/Flask/Django, databases, REST APIs, auth, Docker,
React or custom JS front-ends, tracking, custom training.

## Model

- `yolo11n.pt` (YOLO11 nano), pretrained by Ultralytics on COCO (80 classes).
- Chosen because it is the smallest/fastest YOLO11 model and runs in real time on a CPU.
- Loaded from `models/yolo11n.pt`. If the file is missing, Ultralytics downloads it
  automatically (~5.4 MB) from the official GitHub releases on first run.
- Weights are git-ignored (`models/*.pt`).
- Inference runs on the CPU (the Ultralytics default on this setup).

## Architecture

```text
Webcam → Streamlit UI → OpenCV Frame Capture → Pretrained YOLO → Object Detection
       → Bounding Boxes + Labels + Confidence → Streamlit Display
```

## Main files

| File | Responsibility |
|------|----------------|
| `app.py` | Streamlit page, sidebar controls (input source, confidence slider, camera index, start/stop toggle), browser-webcam mode (`run_browser_webcam`, streamlit-webrtc), local OpenCV webcam loop (`run_webcam`), image-upload mode, stats panel |
| `src/detector.py` | `ObjectDetector` class: loads YOLO, `detect(frame, confidence)` → list of `{box, class_id, label, confidence}`. A `threading.Lock` makes concurrent callers (several viewers) take turns |
| `requirements.txt` / `packages.txt` | Python deps (with the CPU-only PyTorch index) / Debian package `libgl1` + `libglib2.0-0` for OpenCV on Streamlit Cloud |
| `src/utils.py` | `draw_detections`, `format_label`, `calculate_fps`, `count_objects` |
| `models/README.md` | Model choice and download notes |
| `outputs/README.md` | What every screenshot/result image shows + all measured numbers |
| `report/project_report.docx` | Project report |
| `presentation/project_presentation.pptx` | Slides |

## How to run

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
streamlit run app.py
```

Then open <http://localhost:8501> (Streamlit usually opens it automatically) and switch on
**Start detection**.

## Important implementation decisions

- **Three input sources:** "Browser webcam (real-time)" (default, works locally and online),
  "Local webcam (OpenCV)" (works only where the camera is attached to the server, i.e. a local run),
  "Image".
- **Browser webcam (WebRTC):** `webrtc_streamer(mode=SENDRECV, video_frame_callback=process_frame,
  async_processing=True)`. The callback runs in a worker thread: av.VideoFrame → `to_ndarray("bgr24")`
  → detect → draw → `av.VideoFrame.from_ndarray`. `st.*` cannot be called there, so FPS and detections
  are shared through a lock-protected dict in `st.session_state`, and the script polls it every 0.5 s
  while `ctx.state.playing`. With `async_processing`, frames that arrive while YOLO is busy are
  dropped. ICE servers are chosen automatically by streamlit-webrtc: env/secrets `CLOUDFLARE_TURN_KEY_ID`
  and `CLOUDFLARE_TURN_KEY_API_TOKEN`, `TWILIO_ACCOUNT_SID` and `TWILIO_AUTH_TOKEN`, or `HF_TOKEN`,
  otherwise Google STUN. STUN only worked for our cloud test, but some networks will need TURN.
- **Continuous webcam in Streamlit (local OpenCV mode):** a `while True` loop in the script reads frames with
  `cv2.VideoCapture` and updates one `st.empty()` placeholder each frame. When the user
  switches the toggle off (or changes any widget), Streamlit stops the running script and
  starts a new run. The `finally:` block then calls `cap.release()`. This is the simplest
  approach that works without extra packages. It needs the camera on the same machine as the
  Streamlit server, which is fine for a laptop demo. `st.camera_input()` was not used because
  it only captures single snapshots.
- **Model cached** with `@st.cache_resource`, so it is loaded once and not on every rerun.
- **Drawing is done with OpenCV** (`cv2.rectangle`, `cv2.putText`) instead of Ultralytics'
  `result.plot()`, so the image-processing part is visible and easy to explain.
- **Capture size 640x480**: enough for YOLO (which resizes to 640 internally anyway) and fast.
- **FPS** = 1 / (wall-clock time of one loop iteration). No smoothing, no hard-coded values.
- **Object count** = `collections.Counter` over the labels in the current frame (no tracking).
- **Errors:** camera not opening or a frame not being read shows
  "Unable to access webcam. Please check your camera permissions or connected webcam."
  in the video area. An empty scene just shows "No objects above the threshold."
- Each Streamlit browser tab runs its own loop and opens the camera itself. Two tabs at
  once roughly halved the FPS in our tests.

## Measured results (MacBook Air M3, 8 GB, CPU)

- Live webcam (local OpenCV mode): ~22–24 FPS with one tab. Person detected in all 52 saved frames at 0.77–0.94.
- Browser webcam mode, local: works (tested with a Chrome fake camera, 10 fps input, ~10 FPS).
- Browser webcam mode, **online** (Streamlit Cloud, Python 3.14 default): Chrome fake camera with
  30 fps input → mostly 15–24 FPS (single readings down to ~3). Bus sometimes labelled "truck".
- YOLO11n inference: 31.3 ms mean per 640x480 frame (50 runs).
- Full tables (threshold, simulated low light, simulated distance, empty scene) are in
  `outputs/README.md`.

## Limitations

Hardware-dependent speed. Sensitive to lighting, blur and object size/distance. Only the
80 COCO classes. False positives at low thresholds. Per-frame counts only (no tracking).
The OpenCV webcam mode needs the camera on the machine running Streamlit. The online version
runs on a shared free CPU, depends on the viewer's network, and may need a TURN server.
The online real-time mode has not yet been tested with a real (non-fake) camera on the cloud.

## Future scope

Bigger models or a GPU, custom training on specific objects, object tracking,
mobile/edge deployment, ONNX export and other speed optimisations.
