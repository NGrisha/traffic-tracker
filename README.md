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

## Quick Start

By default, the application processes the sample video:

```text
data/videos/traffic_1.mp4
```

### Run with Docker

Build the Docker image and start the application:

```bash
docker compose up --build
```

The application uses the TensorRT engine and NVIDIA GPU. The processed video and tracking history are saved to the `data/output/` directory.

To stop the application, press `Ctrl+C`.

### Run locally (without Docker)

Run the application from the project root:

```bash
python -m src.main
```

Before running locally, update the following lines in `src/main.py`:

* **Line 34:** Comment out the Docker-specific TensorRT engine configuration.
* **Line 35:** Uncomment the local model configuration.
* **Lines 77–86:** Uncomment the visualization code to display the video processing in a window.

Make sure the required Python dependencies are installed and use a model compatible with your local environment.

The processed video and tracking history are saved to the `data/output/` directory.


## Demo

A pre-generated example is available:

```text
data/output/traffic_2_output.mp4
```

Tracking history:

```text
data/output/history_2.json
```
