import os
import cv2
import pandas as pd
from tqdm.auto import tqdm

path_to_video = "data/outputs/video"

for root, dirs, files in os.walk(path_to_video):
    for file in tqdm(files):
        if file.endswith(".mp4"):
            
            num_vid = 1

            video_path = os.path.join(root, file)
            video_name = file.split(".")[0]

            out_vid_path = os.path.join(path_to_video, 'splits', video_name)
            os.makedirs(out_vid_path, exist_ok=True)

            cap = cv2.VideoCapture(video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            full_clip_path = os.path.join(out_vid_path, f'video_{num_vid}.mp4')

            with open('data/outputs/video/splits/names.txt', 'a') as f:
                f.write(f'{full_clip_path}\n')

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_video = cv2.VideoWriter(full_clip_path, fourcc, fps, (width, height))

            for i in range(1, num_frames+1):
                ret, frame = cap.read()
                if not ret:
                    break

                if i % 30 == 0:
                    if ((num_frames+1) - i) < 30:
                        break 
                    num_vid += 1
                    full_clip_path = os.path.join(out_vid_path, f'video_{num_vid}.mp4')
                    out_video = cv2.VideoWriter(full_clip_path, fourcc, fps, (width, height))
                    with open('data/outputs/video/splits/names.txt', 'a') as f:
                        f.write(f'{full_clip_path}\n')
                out_video.write(frame)
            cap.release()
            out_video.release()
            
data_txt = pd.read_table("data/outputs/video/splits/names.txt", header=None)
data_txt = data_txt.sort_values(by=0)
data_txt.to_csv("data/outputs/video/splits/names.csv", index=False)
print("Done")