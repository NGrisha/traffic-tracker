import cv2 
from detector import Detector
import time
import torch
# import pandas as pd
import onnxruntime as ort

ort.preload_dlls()

# benchmark_df = pd.DataFrame(columns=['device', 'fps', 'frames', 'latency_ms'])

SOURCE = 'data/videos/traffic_2.mp4'
warp_up = 50
iteration = 500


def benchmark(device):
    # detector = Detector(conf=0.3, device=device, model_path="models/yolo11n.pt")
    # detector = Detector(conf=0.3, device=device, model_path="models/yolo11n.onnx")
    detector = Detector(conf=0.3, device=device, model_path="models/yolo11n.engine")
    cap = cv2.VideoCapture(SOURCE)

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video FPS: {original_fps}, Width: {width}, Height: {height}")

    latencies = []

    # Warm up the model by running a few iterations
    for _ in range(warp_up):
        ret, frame = cap.read()
        detector.detect(frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # real benchmark
    for _ in range(iteration):
        ret,frame = cap.read()
        if not ret:
            break

        start = time.perf_counter()

        detector.detect(frame)

        if device is not None and device!="cpu":
            torch.cuda.synchronize()  # Wait for GPU to finish

        elapsed = time.perf_counter() - start
        latencies.append(elapsed)

    cap.release()

    frames = len(latencies)
    time_duration = sum(latencies) 
    fps = frames / time_duration 
    avg_latency = 1000 * time_duration / frames 

    # benchmark_df = benchmark_df.append({
    print(f"Device: {device}, FPS: {fps:.2f}, Frames: {frames}, Avg Latency (ms): {avg_latency:.2f}")



benchmark(device=0)