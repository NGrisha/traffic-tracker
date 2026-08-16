from structures import TrackedObjects
import supervision as sv
from utils import detections_to_supervision, supervision_to_tracked_objects

class Tracker:

    def __init__(self):
        self.tracker = sv.ByteTrack()
        self.history = {}


    def update(self, detections, frame_num):

        sv_detections = detections_to_supervision(detections)

        tracked_detections = self.tracker.update_with_detections(sv_detections)

        tracked_objects = supervision_to_tracked_objects(tracked_detections)

        self.update_history(tracked_objects, frame_num=frame_num)

        return tracked_objects


    def update_history(self, tracked_objects, frame_num):

        for obj in tracked_objects:

            if obj.track_id not in self.history:
                self.history[obj.track_id] = {
                    "class_id": int(obj.class_id),
                    "confidence": float(obj.confidence),
                    "trajectory": []
                }

            x1, y1, x2, y2 = obj.bbox

            center = {
                "x": float((x1 + x2) / 2),
                "y": float((y1 + y2) / 2),
                "bbox": [float(x) for x in obj.bbox],
                "frame_num": frame_num,
            }



            self.history[obj.track_id]["trajectory"].append(center)

