import cv2 
import time
import numpy as np

class Visualizer:

    def __init__(self):
        self.colors = {}
        self.last_fps_time = time.time()
        self.fps = 0.0
        self.all_vehicle_ids = set()

    def _get_color(self, track_id):
        if track_id not in self.colors:
            self.colors[track_id] = (
                (track_id * 37) % 256,
                (track_id * 17) % 256,
                (track_id * 97) % 256,
            )
        return self.colors[track_id]

    def draw_detections(self, frame, objects, class_names):

        for obj in objects:
            color = self._get_color(obj.track_id)
            label = class_names[obj.class_id]
            x1, y1, x2, y2 = map(int, obj.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color = color, thickness = 1)
            cv2.circle(frame, ((x1+x2)//2, (y1+y2)//2), 4, color = color, thickness = -1)
            cv2.putText(frame,
                        f"ID:{obj.track_id} c_id:{obj.class_id} name:{label} conf:{obj.confidence:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color = color,
                        thickness = 1),
        return frame

    def trajectory_frame(self, frame, tracked, history):

        active_ids = {track_obj.track_id for track_obj in tracked} 
        self.all_vehicle_ids.update(active_ids)

        for track_id, data in history.items():
            color=self._get_color(track_id)

            if track_id not in active_ids:
                continue

            if len(data["trajectory"]) < 2:
                continue

            x0, y0 = int(data["trajectory"][0]["x"]), int(data["trajectory"][0]["y"])

            for point in data["trajectory"][1:]:
                x, y = int(point["x"]), int(point["y"])
                cv2.line(frame, (x0, y0), (x, y), color=color, thickness=1)
                x0, y0 = x, y

        cv2.putText(frame,
                    f"Vehicles: {len(active_ids)} | Total: {len(self.all_vehicle_ids)}",
                    (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color=(180, 0, 180),
                    thickness=2)

        return frame


    def draw_fps(self, frame):

        current_time = time.time()
        fps = 1.0 / (current_time - self.last_fps_time)
        self.last_fps_time = current_time

        self.fps = 0.9 * self.fps + 0.1 * fps 

        cv2.putText(frame,
                    f"FPS: {self.fps:.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2)
        return frame

    def render(self, frame, tracked_objects, history, class_names):
        frame = self.draw_detections(frame, tracked_objects, class_names)
        frame = self.trajectory_frame(frame, tracked_objects, history)              
        frame = self.draw_fps(frame)
        return frame