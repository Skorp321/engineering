from ultralytics import YOLO

model = YOLO("runs/detect/train2/weights/best.pt")
vid_path = "video/Plechanovo/plechanovo_3.mp4"

results = model.track(vid_path, show=True, tracker="bytetrack.yaml", classes=0)