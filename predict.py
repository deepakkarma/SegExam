from ultralytics import YOLO
model=YOLO("best.pt")
model.predict(source="5.jpg",show=True,save=True,conf=0.5)
