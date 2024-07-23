import argparse
from ultralytics import YOLO
import ffmpegcv
import cv2
import os
from tqdm.auto import tqdm


def main(args):
    folders_of_folders = os.listdir(args.path)
    out_path = 'data/outputs/annos'
    model = YOLO("runs/detect/train2/weights/best.pt")
    for folder_of_folder in folders_of_folders:
        vid_files = os.listdir(os.path.join(args.path, folder_of_folder))
        print(f"Processed {folder_of_folder} folder:")
        for vid_file in tqdm(vid_files):
            print(f'Processed {vid_file} file:')
            vid_path= os.path.join(args.path, folder_of_folder, vid_file)
            vid_name = os.path.split(vid_path)[-1].split(".")[0]
            n_frame = 1
            dict_pathes = {}

            file_output_path = os.path.join(out_path, vid_name)
            os.makedirs(file_output_path, exist_ok=True)

            cap = cv2.VideoCapture(vid_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            scale = height / width

            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

            with tqdm(total=total_frames, desc='Processing frames') as pbar: 
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if n_frame % (2*fps) == 0:
                        result = model.track(frame, persist=True, imgsz=1280, half=True, device=0, classes=0)
                        boxes = result[0].boxes.xywh.cpu().int().tolist()
                        if result[0].boxes.id is None:
                            continue
                        track_ids = result[0].boxes.id.cpu().int().tolist()
                        annotated_frame = result[0].plot()
                        for track_id, box in zip(track_ids, boxes):
                            x, y, w, h = box
                            track_id = int(track_id)
                            if track_id not in dict_pathes.keys():
                                dict_pathes[track_id] = f'{file_output_path}/towel_id_{track_id}.txt'
                            with open(dict_pathes[track_id], 'a') as f:
                                f.write(f"{n_frame} {x} {y} {w} {h}\n")
                        w, h = 900, int(900*scale)
                        scaled_img = cv2.resize(annotated_frame, (w, h))
                        cv2.imshow("Annotate", scaled_img)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    n_frame += 1
                cap.release()
                pbar.update(1)
            pbar.close()
    
if __name__ == "__main__":
    # Создание объекта парсера
    parser = argparse.ArgumentParser(description='Описание программы')
    parser.add_argument('-p', '--path', help='Путь к папке с видео', default='video')

    # Парсинг аргументов командной строки
    args = parser.parse_args()
    main(args)