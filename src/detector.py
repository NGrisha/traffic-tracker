from ultralytics import YOLO
from structures import Detection


class Detector:
    def __init__(self, model_path: str = "models/yolo11n.pt", conf: float = 0.51):
        self.conf = conf
        self.model = YOLO(model_path)
        self.class_names = self.model.names

    def detect(self, frame):
        result = self.model(frame,
                            conf=self.conf,
                            verbose=False,)
        detections = []
        result = result[0]
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            cls = int(box.cls.item())

            conf = float(box.conf.item())

            detections.append(

                Detection(
                    class_id=cls,
                    class_name=self.model.names[cls],
                    confidence=conf,
                    bbox=(x1, y1, x2, y2)
                )

            )

        return detections
    


