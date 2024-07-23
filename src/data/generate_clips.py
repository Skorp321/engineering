import cv2
import numpy as np
import pandas as pd
import os
from tqdm.auto import tqdm
import argparse
import subprocess



def prepare_frame(frame, x, y, w, h, max_width, max_height):
    x1, y1 = int(x) - int(w/2), int(y) - int(h/2) # Верхний левый угол
    x2, y2 = x1 + int(w), y1 + int(h)  # Нижний правый угол
    horizontal_margin_left = int((max_width - w) / 2.0)
    if (2 * horizontal_margin_left) != max_width:
        horizontal_margin_right = horizontal_margin_left + 1
    else:
        horizontal_margin_right = horizontal_margin_left
        
    return cv2.copyMakeBorder(
            frame[y1:y2, x1:x2],
            0,
            max_height - h,
            horizontal_margin_left,
            horizontal_margin_right,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0],
        )


def main():  # sourcery skip: hoist-statement-from-loop
    video_pathes = []
    num_file = 1
    out_path = 'data/outputs/annos'
    out_clip_path = 'data/outputs/video'
    os.makedirs(out_clip_path, exist_ok=True)
    for root, dirs, files in os.walk('video'):
        video_pathes.extend(os.path.join(root, file) for file in files) 
    
    for video_path in tqdm(video_pathes):
        vid_name = os.path.split(video_path)[-1].split(".")[0]
        txt_path = os.path.join(out_path, vid_name)
        files = os.listdir(txt_path)
        for file in files:
            print(f"Processed {file} file.")
            full_path = os.path.join(txt_path, file)
            df  = pd.read_table(full_path, sep=' ', header=None)
            if df.shape[0] < 30:
                continue
            
            cap = cv2.VideoCapture(video_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            clip_name = file.split('.')[0]
            clip_name = f'{clip_name}_{num_file}.mp4'
            full_clip_path = os.path.join(out_clip_path, clip_name)
            print(f'Save vid path: {full_clip_path}')
            with open('data/outputs/video/names.txt', 'a') as f:
                f.write(f'{clip_name}\n')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            out_video = cv2.VideoWriter(full_clip_path, fourcc, fps, (224, 224))

            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            
            df.columns = ['frame', 'x', 'y', 'w', 'h']
            max_width = df['w'].max()
            max_height = df['h'].max()
            df['frame'] = df['frame'].astype(int)
            count = 1  
            with tqdm(total=total_frames, desc='Processing frames') as pbar:         
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if count in df['frame'].values:
                        n, x, y, w, h = df[df['frame']==count].values[0]
                        prepared_frame = prepare_frame(frame, x, y, w, h, max_width, max_height)
                        resized_frame = cv2.resize(prepared_frame, (224, 224))
                        out_video.write(resized_frame)
                        cv2.imshow("YOLOv10 Inference", resized_frame)
                        # Break the loop if 'q' is pressed
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    pbar.update(1)
                    count += 1
            num_file += 1
            cap.release()
            out_video.release()
            cv2.destroyAllWindows()
            pbar.close()
        
if __name__ == '__main__':
    subprocess.run(['python3', 'src/data/generate_txt.py'])
    main()