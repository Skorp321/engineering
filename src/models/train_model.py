from ultralytics import YOLO

# Load YOLOv10n model from scratch
model = YOLO("yolov10l.pt")

# Train the model
model.train(data="data/interim/data.yaml", epochs=100, imgsz=1280, device=0, batch=0.80, optimizer='AdamW')