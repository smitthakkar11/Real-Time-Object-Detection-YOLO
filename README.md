# Real-Time Object Detection using YOLO

A simple real-time object detection system using **YOLO**, **OpenCV** and **Streamlit**.
The webcam is the input. Every frame is passed to a **pretrained** YOLO model,
and the result is shown in a small Streamlit web page with bounding boxes, class names
and confidence scores.

**Live demo:** <https://realtime-object-detection-yolo.streamlit.app>
(choose **Browser webcam (real-time)**, click **START** and allow camera access)

**Course:** Image Processing and Computer Vision

## Team Members

| Name | Roll Number |
|------|-------------|
| Yogesh Dixit | 20233362 |
| Sumitabh Ahirwar | 20233280 |
| Smit Thakkar | 20233570 |
| Vishal Prabhu | 20233582 |


## Description

Object detection means finding *what* objects are in an image and *where* they are.
In this project we built a small application that:

1. gets frames from the webcam, either directly with OpenCV (when the app runs on your
   own laptop) or streamed from your browser with WebRTC (works locally and online),
2. processes the frames one by one,
3. runs each frame through a pretrained YOLO model (Ultralytics **YOLO11n**),
4. draws a box, the object name and the confidence score for every detected object,
5. shows the processed frame, the FPS and a simple object count in a Streamlit page.

> **Academic honesty note:** we did **not** train YOLO and we did not design any new
> detection algorithm. A pretrained YOLO model (trained by Ultralytics on the COCO
> dataset) is used for detection. Our work is integrating that model with OpenCV
> and a simple Streamlit interface for real-time detection.

## Features

- Real-time object detection from the webcam
- Two webcam modes: **Browser webcam (real-time)** via WebRTC, which also works in the online
  version, and **Local webcam (OpenCV)**, which reads the camera directly with `cv2.VideoCapture`
- Deployed online on Streamlit Community Cloud
- Pretrained YOLO11n model (80 COCO classes such as person, bottle, laptop, cell phone, cup, chair)
- Bounding boxes drawn with OpenCV
- Class label and confidence score on every box (for example `person 0.92`)
- Simple Streamlit interface with a start/stop switch
- Adjustable confidence threshold
- FPS calculated from the actual processing time of each frame
- Simple per-frame object count (for example `person: 2`, `bottle: 1`)
- Optional image upload mode (for testing without a webcam)
- Clear error message if the webcam cannot be opened

## Technologies Used

| Technology | Used for |
|------------|----------|
| Python | Programming language |
| OpenCV (`opencv-python`) | Webcam capture, drawing boxes and text, colour conversion |
| Ultralytics YOLO (`ultralytics`) | Loading the pretrained YOLO11n model and running detection |
| Streamlit | Very basic web user interface |
| streamlit-webrtc | Streams the viewer's browser webcam to the app and the processed video back (WebRTC), for real-time detection online |
| NumPy | Converting an uploaded image file into an OpenCV image |

There is no separate backend server, database, REST API or custom JavaScript front-end.
Streamlit is the only UI framework. `streamlit-webrtc` is a ready-made Streamlit component.

## Project Structure

```text
Real-Time-Object-Detection-YOLO/
├── README.md               <- this file
├── PROJECT_CONTEXT.md      <- short technical summary of the project
├── requirements.txt        <- Python packages
├── packages.txt            <- system library needed by OpenCV on Streamlit Cloud (Linux)
├── .gitignore
├── app.py                  <- Streamlit UI + webcam loop
├── src/
│   ├── __init__.py
│   ├── detector.py         <- loads YOLO and runs detection
│   └── utils.py            <- drawing, FPS, object counting
├── models/
│   └── README.md           <- where/how the model weights are downloaded
├── outputs/
│   ├── README.md           <- description of every output + measured numbers
│   ├── screenshots/        <- screenshots of the running app
│   └── sample_results/     <- annotated result images and test outputs
├── report/
│   └── project_report.docx
└── presentation/
    └── project_presentation.pptx
```

## Installation

You need **Python 3.10, 3.11 or 3.12** (we tested with Python 3.11.9).

**1. Open a terminal in the project folder**

```bash
cd Real-Time-Object-Detection-YOLO
```

**2. Create a virtual environment**

```bash
python -m venv venv
```

**3. Activate it**

Windows:

```bash
venv\Scripts\activate
```

macOS / Linux:

```bash
source venv/bin/activate
```

**4. Install the dependencies**

```bash
pip install -r requirements.txt
```

This installs Streamlit, streamlit-webrtc, OpenCV and Ultralytics. Ultralytics also installs
PyTorch, which is a large download (a few hundred MB), so the first install can take a few minutes.
`requirements.txt` points pip to the CPU-only PyTorch builds, because no GPU is needed.

## Run

```bash
streamlit run app.py
```

Streamlit starts a small local server and opens the app in your browser
(usually at <http://localhost:8501>). Everything runs on your own laptop.
To stop the server, press `Ctrl + C` in the terminal.

## Model

- **Model used:** `yolo11n.pt` (YOLO11 *nano*), pretrained on the COCO dataset (80 classes).
- **Why a lightweight model:** the nano version is the smallest and fastest YOLO11 model.
  It runs in real time on a normal laptop CPU without a GPU. Bigger versions are more
  accurate but slower.
- **How it is downloaded:** automatically. On the first run, Ultralytics sees that
  `models/yolo11n.pt` does not exist and downloads it (about 5.4 MB) from the official
  Ultralytics GitHub releases. This needs an internet connection only once.
- **Where it is stored:** `models/yolo11n.pt`. The file is in `.gitignore`, so it is not
  committed to GitHub.

See [models/README.md](models/README.md) for more details.

## Usage

1. Start the app with `streamlit run app.py`, or open the
   [live demo](https://realtime-object-detection-yolo.streamlit.app).
2. In the sidebar, choose an **Input source**:
   - **Browser webcam (real-time)**: click **START** and allow camera access in the browser.
     Works on your laptop and in the online version.
   - **Local webcam (OpenCV)**: switch on **Start detection**. The app opens the camera itself,
     so this only works when the app runs on the same computer as the webcam. The first time,
     your operating system may ask for permission to use the camera. Allow it.
   - **Image**: upload a `.jpg`/`.png` to run detection on a single photo.
3. Point the webcam at some objects (people, bottle, cup, phone, laptop, book ...).
   Boxes, names and confidence scores appear on the video. FPS and the object count
   are shown on the right.
4. Move the **Confidence threshold** slider to hide weak detections (higher value)
   or show more detections (lower value).
5. Click **STOP** (browser webcam) or switch off **Start detection** (local webcam) to stop.
   The webcam is released.

If you have more than one camera, use **SELECT DEVICE** (browser webcam) or change
**Camera index** (local webcam; 0 is usually the built-in webcam).

## How it works

```text
Webcam → Streamlit UI → OpenCV Frame Capture → Pretrained YOLO → Object Detection
       → Bounding Boxes + Labels + Confidence → Streamlit Display
```

- `app.py` opens the camera with `cv2.VideoCapture`, reads a frame, calls the detector,
  draws the results and updates an `st.empty()` placeholder with the new frame. This
  repeats in a loop, which gives a continuous video.
- `src/detector.py` runs `model(frame, conf=threshold)` and turns the result into a simple
  list of `{box, class_id, label, confidence}` dictionaries.
- `src/utils.py` draws the boxes and labels with `cv2.rectangle` / `cv2.putText`,
  calculates `FPS = 1 / (time for one loop)` and counts objects per class.
- OpenCV gives frames in **BGR** colour order, so each frame is converted to **RGB**
  before Streamlit displays it.

**Two ways to get webcam frames:**

- **Local webcam (OpenCV):** `cv2.VideoCapture` reads the webcam of the computer where
  `streamlit run` is running. This is the simplest method and is what our first tests used.
  It cannot work in the online version, because the cloud server has no webcam.
- **Browser webcam (real-time):** the `streamlit-webrtc` component asks the browser for the
  camera and streams the video to the app over WebRTC. For every frame it calls our
  `process_frame()` function in a separate thread. That function converts the frame to an
  OpenCV image, runs YOLO, draws the boxes and sends the frame back to the browser. Because
  `st.*` functions cannot be used inside that thread, the latest FPS and detections are stored
  in a small dictionary protected by a lock, and the page reads them twice per second.

We did **not** use Streamlit's `st.camera_input()`, because that widget only takes
single snapshots and cannot give a continuous video stream.

**Note on object counting:** the count is a simple *per-frame* count of the current
detections. It does not track objects between frames, so the same person walking
past the camera is not counted as "one unique person".

## Online Deployment (Streamlit Community Cloud)

The app is deployed at **<https://realtime-object-detection-yolo.streamlit.app>**, straight from
this GitHub repository (branch `main`, main file `app.py`). Every push to `main` redeploys it.

Things that were needed for the cloud (Linux) server:

- `requirements.txt` uses the CPU-only PyTorch index. Otherwise pip installs the much larger CUDA build.
- `packages.txt` installs `libgl1` and `libglib2.0-0`, which OpenCV needs on Linux.
- The model weights are downloaded automatically on the first start, just like locally.

**WebRTC and TURN servers:** the browser and the cloud server need a network path for the video.
By default `streamlit-webrtc` uses Google's public STUN server, and that worked in our test.
Some networks, for example strict college or office firewalls, also need a **TURN** relay server.
If the browser webcam stays on "connecting", add one of these as app **Secrets** in the Streamlit Cloud
settings. `streamlit-webrtc` picks them up automatically, and no code change is needed:

```toml
# free Hugging Face account -> Settings -> Access Tokens
HF_TOKEN = "hf_..."
# or Cloudflare Realtime TURN
# CLOUDFLARE_TURN_KEY_ID = "..."
# CLOUDFLARE_TURN_KEY_API_TOKEN = "..."
```

**Privacy:** in the browser webcam mode, video frames are sent to the server that runs the app,
processed in memory and sent back. The app does not save any frames.

## Results

Measured on our test laptop (MacBook Air, Apple M3, 8 GB RAM, CPU only):

- The app detected a person in the live webcam feed with confidence between **0.77 and 0.94** (52 saved frames).
- The FPS shown in the app was about **22–24 FPS** with one browser tab open. It dropped to
  about **10–14 FPS** while two browser tabs were running detection at the same time.
- **Online version (Streamlit Community Cloud), browser webcam mode:** we tested it with Chrome's
  fake camera playing a 30 fps video made from the sample images. The app mostly showed
  **15–24 FPS**, with single readings as low as 3 when frames arrived unevenly over the network. It
  detected the bus and people and showed 0 objects on the empty frames. At that speed the bus was
  sometimes labelled "truck".
- YOLO11n inference alone took about **31 ms** per 640x480 frame (average of 50 runs).
- On the Ultralytics sample image `bus.jpg`, the app found 1 bus and 4 persons at threshold 0.50.
- Lowering the threshold to 0.25 showed extra, less reliable detections. On `zidane.jpg` a tie
  appeared (0.46). In the webcam feed a wristwatch appeared as `clock 0.33` and once was
  wrongly labelled `bottle 0.25`.

Your numbers will be different on other hardware. All screenshots, result images and the full
tables are in [outputs/](outputs/README.md).

## Limitations

- Speed depends on the laptop. Without a GPU, older laptops may get a low FPS.
- Webcam quality and motion blur affect the results.
- Low lighting reduces detection quality.
- Small or far-away objects are harder to detect.
- The pretrained model only knows the 80 COCO classes. Other objects are either
  missed or given the wrong label.
- Low thresholds can give false positives (e.g. a watch labelled as a bottle).
- The count is per frame only. There is no tracking.
- The **Local webcam (OpenCV)** mode needs the webcam on the same computer that runs
  `streamlit run app.py`. Online, use the browser webcam mode.
- The online version runs on a shared, free cloud CPU. Speed depends on the server load and on
  the viewer's network, and some networks need a TURN server (see above).
- Each browser tab that runs detection uses its own video stream. All viewers share one model,
  which processes one frame at a time.

## Troubleshooting

- **"Unable to access webcam"**: close other apps that use the camera (Zoom, Teams,
  Photo Booth, ...), check the **Camera index**, and check camera permissions:
  - Windows: *Settings → Privacy & security → Camera* → allow desktop apps.
  - macOS: *System Settings → Privacy & Security → Camera* → allow the terminal app you
    used to run `streamlit run app.py` (for example Terminal or VS Code).
- **Very low FPS**: close other browser tabs that are running the app. Each tab runs its
  own detection loop.
- **Browser webcam stuck on "connecting"** (online): your network probably blocks direct WebRTC
  connections. Add a TURN secret as described in *Online Deployment*.
- **First run is slow**: the model weights are being downloaded, and PyTorch is loading.

## Future Scope

- Try bigger pretrained models (YOLO11s/m) or use a GPU for better accuracy
- Custom training for a specific set of objects
- Object tracking (unique IDs across frames)
- Mobile or edge-device deployment (e.g. Raspberry Pi, Jetson)
- Speed optimisations such as model export (ONNX) and lower input resolution

## Credits

- The object detection model is the pretrained **YOLO11n** from [Ultralytics](https://github.com/ultralytics/ultralytics)
  (licensed under AGPL-3.0). We did not train or modify it.
- `bus.jpg` and `zidane.jpg`, used for the image-mode tests in `outputs/`, are sample images that ship with the
  Ultralytics package.

## References

1. Ultralytics YOLO documentation: <https://docs.ultralytics.com/>
2. Ultralytics YOLO11 model page: <https://docs.ultralytics.com/models/yolo11/>
3. OpenCV documentation: <https://docs.opencv.org/>
4. Streamlit documentation: <https://docs.streamlit.io/>
5. J. Redmon, S. Divvala, R. Girshick, A. Farhadi, "You Only Look Once: Unified, Real-Time Object Detection", CVPR 2016.
6. T.-Y. Lin et al., "Microsoft COCO: Common Objects in Context", ECCV 2014.
