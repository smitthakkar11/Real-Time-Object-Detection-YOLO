# Model Weights

This project uses the pretrained **YOLO11n** ("nano") model from Ultralytics.
We did **not** train this model. It was trained by Ultralytics on the
**COCO** dataset, which has 80 everyday object classes (person, bottle,
laptop, cell phone, cup, chair, car, and so on).

## How the weights are downloaded

You do not need to download anything by hand.

- `src/detector.py` loads the model from `models/yolo11n.pt`.
- If that file is not there yet, the Ultralytics library **automatically
  downloads** it from the official Ultralytics GitHub releases
  (`https://github.com/ultralytics/assets/releases`) the first time the app runs.
- The file is saved in this `models/` folder, so later runs work offline.
- The file is about **5.4 MB**.

The first run therefore needs an internet connection. After that, the app
loads the weights from this folder.

## Why the weights are not in the Git repository

`.pt` files are listed in `.gitignore`. Weight files are binary files that
do not belong in source control, and the automatic download already
handles them, so there is no need to commit them.

## Why the nano model?

Ultralytics offers several sizes of the same model (n, s, m, l, x).
The nano model is the smallest and fastest one. It is a little less accurate
than the bigger models, but it runs in real time on a normal laptop CPU without
a GPU, which is what this project needs.

On our test laptop (MacBook Air, Apple M3, 8 GB RAM, CPU only), one
640x480 frame took about **31 ms** to process with YOLO11n
(average of 50 runs).

## Using a different model

To try another pretrained model, change `DEFAULT_MODEL` in `src/detector.py`,
for example:

```python
DEFAULT_MODEL = "yolov8n.pt"   # older YOLOv8 nano
DEFAULT_MODEL = "yolo11s.pt"   # bigger YOLO11 "small": more accurate, slower
```

Ultralytics will download the new weights automatically in the same way.
