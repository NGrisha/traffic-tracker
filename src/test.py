from detector import Detector 
import os 

detector = Detector(model_path="models/yolo11n_81cls.pt")
print(detector.class_names)
