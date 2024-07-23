import time
import numpy as np
import streamlit as st
import cv2
from ultralytics import YOLO
import ffmpegcv


def main():
    # Настройка заголовка и и описания
    st.title("Streamlit Video Stream")
    st.text("Показ видео с помощью Streamlit и opencv")

    # Путь до видео
    vid_path = "/home/skorp321/Projects/engineering/video/Plechanovo/plechanovo2_0.mp4"

    # Load the model
    model = YOLO("runs/detect/train2/weights/best.pt")
    cap = cv2.VideoCapture(vid_path)
    #cap = ffmpegcv.VideoCapture(vid_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    n_frame = 1
    #Создает область для показа видео
    frame_window = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.write("Can't receive frame (stream end?). Exiting...")
            break
        if n_frame % fps == 0:
            cv2.imshow('frame', frame)
            start = time.time()
            result = model.track(frame, persist=True, imgsz=1280, half=True, device=0, classes=0)
            
            boxes = result[0].boxes.xyxy.cpu().int().tolist()   
            track_id = result[0].boxes.id.cpu().int().tolist()
            annotated_frame = result[0].plot()
            copy_frame = frame.copy()
            for box, track_id in zip(boxes, track_id):
                x1,y1, x2, y2 = box
                track_id = int(track_id)
                for id in track_id:
                    name_file = f'towel_track_{id}.mp4'
                    
                    
            
            end = time.time() - start
            cv2.putText(annotated_frame, f"FPS: {round(1 / end, 2)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            
            frame_window.image(result, channels="BGR")  
        n_frame += 1
        
    cap.release()
if __name__ == "__main__":
    main()