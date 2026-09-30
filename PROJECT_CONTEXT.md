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
- NumPy: decoding uploaded image bytes

Deliberately **not** used: FastAPI/Flask/Django, databases, REST APIs, auth, Docker,
React or other JS front-ends, WebSockets/streamlit-webrtc, tracking, custom training.

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
| `app.py` | Streamlit page, sidebar controls (input source, confidence slider, camera index, start/stop toggle), webcam loop, image-upload mode, stats panel |
| `src/detector.py` | `ObjectDetector` class: loads YOLO, `detect(frame, confidence)` → list of `{box, class_id, label, confidence}` |
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

- **Continuous webcam in Streamlit:** a `while True` loop in the script reads frames with
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

- Live webcam: ~22–24 FPS with one tab. Person detected in all 52 saved frames at 0.77–0.94.
- YOLO11n inference: 31.3 ms mean per 640x480 frame (50 runs).
- Full tables (threshold, simulated low light, simulated distance, empty scene) are in
  `outputs/README.md`.

## Limitations

Hardware-dependent speed. Sensitive to lighting, blur and object size/distance. Only the
80 COCO classes. False positives at low thresholds. Per-frame counts only (no tracking).
The camera must be local to the machine running Streamlit.

## Future scope

Bigger models or a GPU, custom training on specific objects, object tracking,
mobile/edge deployment, ONNX export and other speed optimisations.
