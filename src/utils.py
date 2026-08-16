import numpy as np
import supervision as sv
from structures import TrackedObjects
import json

def detections_to_supervision(detections):

    if not detections:
        return sv.Detections.empty()

    xyxy = np.array([d.bbox for d in detections], dtype=np.float32)
    confidences = np.array([d.confidence for d in detections], dtype=np.float32)
    class_ids = np.array([d.class_id for d in detections], dtype=np.int32)


    return sv.Detections( 
        xyxy=np.array(xyxy),
        confidence=np.array(confidences),
        class_id=np.array(class_ids)
    )

def supervision_to_tracked_objects(tracked_objects):

    if len(tracked_objects) == 0:
        return []

    tracked_obj = []

    for i in range(len(tracked_objects)):
        tracked_obj.append(
            TrackedObjects(
                track_id=int(tracked_objects.tracker_id[i]),
                class_id=int(tracked_objects.class_id[i]),
                confidence=float(tracked_objects.confidence[i]),
                bbox=tuple(tracked_objects.xyxy[i])
            )
        )

    return tracked_obj


def save_history_to_json(history, filename):

    with open(filename, "w") as f:
        json.dump(history, f, indent=4)



# def save_history_to_json(tracked, filename):
#     history = {}

#     for obj in tracked:
#         # history[obj.track_id] = {
#         #     "class_id": int(obj.class_id),
#         #     "confidence": float(obj.confidence),
#         #     "center": [float((obj.bbox[0] + obj.bbox[2]) / 2), float((obj.bbox[1] + obj.bbox[3]) / 2)],
#         #     "bbox": [float(x) for x in obj.bbox]
#         # }

#         if obj.track_id not in history:
#             history[obj.track_id] = {
#                 "trajectory": []
#             }

#         x1, y1, x2, y2 = obj.bbox

#         center = [
#             float((x1 + x2) / 2),
#             float((y1 + y2) / 2)
#         ]

#         history[obj.track_id]["trajectory"].append({
#             "class_id": int(obj.class_id),
#             "confidence": float(obj.confidence),
#             "center": center
#         })

#     with open(filename, 'w') as f:
#         json.dump(history, f)

