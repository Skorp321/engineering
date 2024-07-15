import cv2
import time
from ultralytics import YOLO
import ffmpegcv

# Load a model
model = YOLO("runs/detect/train2/weights/best.pt")  # pretrained YOLOv8n model

# Open the video file
video_path = "video/Plechanovo/plechanovo2_0.mp4"
#cap = cv2.VideoCapture(video_path)
cap = ffmpegcv.VideoCapture(video_path)

output_path = 'black_river_tr.mp4'  # Путь и имя для сохранения видео
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Тип кодека (например, MPEG-4)
fps = cap.fps
print(f"fps: {fps}")
#fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_size = (640, 480)  # Размер кадра

#out = cv2.VideoWriter('black_river.avi', cv2.VideoWriter_fourcc( * 'XVID'), fps, frame_size)
out = ffmpegcv.VideoWriterNV(output_path, "h264", fps)
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if not success:
        # Break the loop if the end of the video is reached
        break

    start = time.time()
    # Run YOLOv8 inference on the frame
    results = model.track(frame, persist=True, imgsz=1280, half=True, device=0, classes=0)
    #results = model(frame, imgsz=1280, half=True, device=0, save=True)  # return a list of Results objects
    boxes = results[0].boxes.xywh.cpu()
    track_ids = results[0].boxes.id.int().cpu().tolist()
    annotated_frame = results[0].plot()
    for box, track_id in zip(boxes, track_ids):
        x, y, w, h = box
    # Visualize the results on the frame
    annotated_frame = results[0].plot()
    end = time.time() - start
    cv2.putText(annotated_frame, f"fps: {int(1 / end)}", (20, 50), 1, 1.5, (0,0,0), 2, -1)
    # Display the annotated frame

    cv2.imshow("YOLOv10 Inference", annotated_frame)
    out.write(annotated_frame)
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
# Release the video capture object and close the display window

cap.release()
out.release()
cv2.destroyAllWindows()

print("Done!")