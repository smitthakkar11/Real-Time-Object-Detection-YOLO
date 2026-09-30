# Outputs

Every image in this folder was produced by the application code in this
repository. None of them were edited or mocked up.

**Test laptop:** MacBook Air, Apple M3, 8 GB RAM, macOS 26.5.1, Python 3.11.9,
Ultralytics 8.4.167, PyTorch 2.14.0 (CPU), OpenCV 5.0.0, Streamlit 1.64.0.
Model: pretrained `yolo11n.pt`.

## screenshots/

Browser screenshots of the Streamlit app while it was running.

| File | What it shows |
|------|---------------|
| `01_home_screen.png` | App after start-up, before detection is switched on |
| `02_webcam_live_detection.png` | Live webcam detection: `person 0.88`, FPS 23.4 |
| `03_webcam_live_detection_2.png` | Live webcam detection: `person 0.83`, FPS 20.6 |
| `04_image_mode_multiple_objects.png` | Image mode, Ultralytics sample `bus.jpg`: 1 bus + 4 persons |
| `05_image_mode_conf_050.png` | Image mode, sample `zidane.jpg`, threshold 0.50: 2 persons |
| `06_image_mode_conf_025.png` | Same image, threshold 0.25: 2 persons + `tie 0.46` |
| `07_empty_scene_no_objects.png` | Crop of a webcam frame showing only wall/ceiling: 0 objects, app keeps working |
| `08_webcam_unavailable_error.png` | Error message when the camera could not be opened (camera permission was not granted) |

## sample_results/

| File | What it shows |
|------|---------------|
| `webcam_person_far.png`, `webcam_person_near.png` | Live webcam frames at two distances: `person 0.93` (farther) and `person 0.94` (closer) |
| `webcam_conf025_watch_detected_as_clock.png` | Live frame at threshold 0.25: a wristwatch detected as `clock 0.33` |
| `webcam_conf025_watch_detected_as_bottle.png` | Live frame at threshold 0.25: the same watch wrongly labelled `bottle 0.25` (false positive) |
| `bus_detected.jpg`, `zidane_detected.jpg` | Detections on the two sample images that ship with Ultralytics |
| `threshold_comparison_zidane.jpg` | Same image at thresholds 0.25 / 0.50 / 0.80 |
| `low_light_simulation_bus.jpg` | **Simulated** low light (pixel values scaled to 100%, 25%, 10%, 5%) |
| `distance_simulation_bus.jpg` | **Simulated** distance (image shrunk to x1, x0.5, x0.25, x0.125 on a grey canvas) |

`bus.jpg` and `zidane.jpg` are the sample images that ship with the Ultralytics package.

## Measured numbers

**FPS shown in the app during live webcam detection** (FPS = 1 / time for one
loop iteration: read frame + YOLO + drawing + sending to the browser):

- About **22–24 FPS** when one browser tab was running detection.
- About **10–14 FPS** while a second browser tab was running detection at the
  same time. Both tabs were sharing the camera and the CPU.

**YOLO inference alone** on a 640x480 frame (CPU, 50 runs after warm-up):
mean **31.3 ms**, min 29.3 ms, max 34.0 ms.

**Confidence threshold** (number of detections):

| Image | 0.25 | 0.50 | 0.80 |
|-------|------|------|------|
| bus.jpg | 5 (bus, 4 person) | 5 (bus, 4 person) | 4 (bus, 3 person) |
| zidane.jpg | 3 (2 person, tie 0.46) | 2 (2 person) | 1 (person 0.84) |

**Simulated low light** on bus.jpg (threshold 0.50):

| Brightness | Mean pixel value | Detections |
|------------|------------------|------------|
| 100% | 117.1 | 5: bus 0.94, person 0.89 / 0.88 / 0.86 / 0.62 |
| 50% | 58.6 | 5: bus 0.94, person 0.88 / 0.87 / 0.86 / 0.62 |
| 25% | 29.3 | 5: bus 0.95, person 0.89 / 0.87 / 0.86 / 0.55 |
| 10% | 11.7 | 4: bus 0.90, person 0.89 / 0.87 / 0.83 |
| 5% | 5.8 | 3: person 0.86 / 0.84 / 0.64 (bus missed) |

**Simulated distance** on bus.jpg (threshold 0.50):

| Scale | Detections |
|-------|------------|
| x1.0 | 5: bus 0.94, 4 persons |
| x0.5 | 4: bus 0.88, 3 persons |
| x0.25 | 4: bus 0.84, 3 persons (0.79 / 0.59 / 0.58) |
| x0.125 | 0: nothing detected |

**Empty scene:** an all-black 640x480 frame and a wall-only crop from the
webcam both gave 0 detections at 0.50.

The low-light and distance tests are **simulations** made with OpenCV
(brightness scaling and resizing). They are not real low-light or
long-distance camera recordings.
