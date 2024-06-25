import cv2
import supervision as sv
from ultralytics import YOLO

cap = cv2.VideoCapture(0)
model=YOLO("best.pt")
box_annotator  = sv.BoxAnnotator (
      thickness=2,
      text_thickness=2,
      text_scale=1
)
while True:
        ret, frame =cap.read()

        result = model(frame)[0]
        detections = sv.Detections.from_yolov8(result)
        labels = [
f"{model.model.names[class_id]}{confidence:0.2f}"
for _,confidence,class_id,_
in detections
                ]
        frame = box_annotator.annotate(scene=frame,detections=detections, labels=labels)
        cv2.imshow("yolov8m",frame)
        if(cv2.waitKey (30)==27):
            break
