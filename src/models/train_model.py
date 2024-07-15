from ultralytics import YOLO
import clearml

# Load YOLOv10n model from scratch
model = YOLO('runs/detect/train/weights/best.pt')

clearml.browser_login()
# Train the model
model.train(data="data/interim/data.yaml", 
            epochs=100, 
            imgsz=1280, 
            device=0, 
            batch=4,
            mosaic=0.5,
            mixup=0.2,
            patience=10,
            cache=True,
)

metrics = model.val()  # no arguments needed, dataset and settings remembered
print(f"map50-95: {metrics.box.map}")  # map50-95
print(f"map50: {metrics.box.map50}")  # map50
print(f"map70: {metrics.box.map75}")  # map75
print(f"map: {metrics.box.maps}") 