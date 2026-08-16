import cv2
from detector import Detector
from tracker import Tracker
from visualizer import Visualizer
from utils import save_history_to_json
import os
import json
# import re
# from pathlib import Path

conf = float(os.getenv("YOLO_CONF", "0.3"))

# videos_num = [
#     int(match.group(1))
#     for file in Path("data/videos").glob("traffic_*.mp4")
#     if (match := re.fullmatch(r"traffic_(\d+)\.mp4", file.name))
# ]

# video_number = max(videos_num, default=0) + 1

video_number = 1
LIVE_CAMERA = False

if LIVE_CAMERA:
    SOURCE = 0
else:
    SOURCE = f"data/videos/traffic_{video_number}.mp4" # ex. "data/videos/traffic_NUM.mp4"
# VIDEO_PATH =  f"data/videos/traffic_{video_number}.mp4"  

HISTORY_CENTER_PATH = f"data/output/history_{video_number}.json"
VIDEO_OUTPUT_PATH = f"data/output/traffic_{video_number}_output.mp4"

detector = Detector(conf=conf)
tracker = Tracker()
visualizer = Visualizer()


with open(HISTORY_CENTER_PATH, "w") as f:
    json.dump({}, f)

def main():

    cap = cv2.VideoCapture(SOURCE)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video FPS: {fps}, Width: {width}, Height: {height}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(VIDEO_OUTPUT_PATH, fourcc, fps, (width, height))

    if not cap.isOpened():
        raise RuntimeError("Cannot open video.")

    frame_num = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_num += 1

        detections = detector.detect(frame)
        tracked = tracker.update(detections, frame_num=frame_num)

        # print(tracked)
        frame = visualizer.render(frame, tracked, tracker.history, detector.class_names)
        out.write(frame)


        
        cv2.imshow("Traffic Tracker", frame)
        


        key = cv2.waitKey(1)

        if key == ord("q"):
            save_history_to_json(
            tracker.history,
            HISTORY_CENTER_PATH
        )
            
            break

    save_history_to_json(
    tracker.history,
    HISTORY_CENTER_PATH)
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()