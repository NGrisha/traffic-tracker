# Object Detection and Multi-Object Tracking

Object detection and multi-object tracking pipeline based on **YOLO11** and **ByteTrack**.

## Features

* YOLO11 object detection
* Multi-object tracking with ByteTrack
* Bounding boxes with persistent tracking IDs
* Object trajectories, FPS, and tracked object count visualization
* Tracking history and output video saving
* Inference benchmarks: PyTorch vs ONNX vs TensorRT

## Inference Benchmarks

Tested on **NVIDIA GeForce GTX 1650 Max-Q** using 500 frames.

| Backend  | Device | FPS    | Avg Latency (ms) |
| -------- | ------ | ------ | ---------------: |
| PyTorch  | CPU    | 19.13  |            52.28 |
| ONNX     | CPU    | 15.24  |            65.60 |
| PyTorch  | GPU    | 76.45  |            13.08 |
| ONNX     | GPU    | 61.31  |            16.31 |
| TensorRT | GPU    | 108.09 |             9.25 |

### Quick Start

By default, the application processes `data/videos/traffic_1.mp4` using the Linux TensorRT engine for GPU inference (line 34 in `src/main.py`).

**Run with Docker:**

```bash
docker compose up --build
```

**Run locally:**

```bash
python -m src.main
```

To use a different configuration, uncomment the corresponding line in `src/main.py` and comment out the others:

* **CPU:** line 33 — `models/yolo11n.pt`
* **Linux / Docker (GPU):** line 34 — `models/yolo11n.engine` *(default)*
* **Windows (GPU):** line 35 — `models/yolo11n_win.engine`

To display video processing in a window, uncomment lines 77–86.

Output videos and tracking history are saved to `data/output/`.

## Demo

A pre-generated example is available:

```text
data/output/traffic_2_output.mp4
```

Tracking history:

```text
data/output/history_2.json
```
