from ultralytics import YOLO 

model = YOLO("models/yolo11n.pt")


# model.export(format="engine", 
#              imgsz=640, 
#              batch=1, 
#              device=0, 
#              half=True,) 


# model.export(format="onnx", 
#              imgsz=640, 
#              batch=1, 
#              device="cpu", )