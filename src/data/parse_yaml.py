#!/usr/bin/env python
# coding: utf-8

import random
import re
import xml.etree.ElementTree as ET
from PIL import Image
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import yaml
import cv2
from tqdm.auto import tqdm
import shutil
import subprocess

def procesed_files(anno_path, output_path, anno_dict):
    files  = os.listdir(anno_path)    

    for file in files:
        
        print(f"Collecting data {file}!")
        num = 1
        absolute_path = "/home/skorp321/Projects/engineering"
        #file_mod = file.replace('_', "")
        result = re.sub(r'\d.*', '', file)
        if result.endswith('_'):
            result = result[:-1]
        folder_name = result.capitalize()
        
        video_path = anno_path.replace('data', 'video').replace('anno', folder_name)
        video_path = os.path.join(video_path, f"{file}.mp4")
        
        cap = cv2.VideoCapture(video_path)
        frame_count  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(video_path)
        with tqdm(total=frame_count) as pbar:
            while cap.isOpened():
                ret, frame = cap.read()

                if not ret:
                    break
                
                img_folder = os.path.join(output_path, folder_name,  'images')
                os.makedirs(img_folder, exist_ok=True)
                cv2.imwrite(os.path.join(img_folder, f"{num:06d}.jpg"), frame)
                num += 1
                pbar.update(1)

        cap.release()
        cv2.destroyAllWindows()

        anno_path_full = os.path.join(anno_path, file, 'annotations.xml')

        # Загрузим XML файл
        tree = ET.parse(anno_path_full)

        # Получим корневой элемент XML файла
        root = tree.getroot()

        # Find the "width" and "height" elements
        width_element = root.find('.//width')
        height_element = root.find('.//height')

        # Extract the values
        width = int(float(width_element.text))
        height = int(float(height_element.text))

        # Проитерируемся по дочерним элементам корня
        for track in root.findall("track"):
            track_id = track.get("id")
            label = track.get("label")

            # Переберем все baundingbox`s
            for box in track.findall("box"):
                
                occluded = int(float(box.get("occluded")))
                
                if not occluded:
                            
                    frame  = int(float(box.get("frame"))) +1
                    
                    xtl = int(float(box.get("xtl")))
                    ytl = int(float(box.get("ytl")))
                    xbr = int(float(box.get("xbr")))
                    ybr = int(float(box.get("ybr")))

                    x_dif = np.abs(xtl - xbr)
                    y_dif = np.abs(ytl - ybr)

                    xw = (xbr - xtl) / width
                    yh = (ybr - ytl) / height
                    xc = (xbr + xw / 2) / width
                    yc = (ytl + yh / 2) / height
                    

                    file_name = f"{frame:06d}.txt"
                    output_path_full = os.path.join(output_path, folder_name, 'labels')
                    os.makedirs(output_path_full, exist_ok=True)
                    res_path = os.path.join(output_path_full, file_name) 
                    
                    # Open the file in append mode (create new or append)
                    with open(res_path, 'a+') as file:
                        # Write the string
                        key = [key for key, val in anno_dict.items() if val == label]
                        string_to_write = f"{key[0]} {xc} {yc} {xw} {yh}"
                        file.write(string_to_write + '\n')
    print()
    
def prepare_files(anno_path, anno_dict):
    print('Prepare fiels!')
    data = {}
    count = 1
    all_files = []
    folders  = os.listdir(anno_path)
    
    for folder in folders:
        files = os.listdir(os.path.join(anno_path, folder, 'images'))
        
        for file in files:
            all_files.append(os.path.join(anno_path, folder,  'images',  file))
            
    random.shuffle(all_files)
    
    train, test = all_files[:int(len(all_files) * 0.8)], all_files[int(len(all_files)  *  0.8):]
    
    for item in tqdm(train):
        copy_fiels(item, output_path, 'train', count)
        count += 1
        
    count =  1
    for item in tqdm(test):
        copy_fiels(item, output_path, 'val', count)
        count += 1
    
    for folder in folders:
        shutil.rmtree(os.path.join(output_path, folder))
        
    data['path'] = output_path
    data['train'] = 'train'
    data['val']  = 'val'
    
    data['nc'] = len(anno_dict.keys())
    data['names'] = anno_dict
    
    with open(os.path.join(output_path,  'data.yaml'),  'w') as file:
        yaml.dump(data,  file, default_flow_style=False)
    
    print('Done!')

def copy_fiels(item, output_path, kaind, count):
    name = f"{count:06d}.jpg"
    folder_path_img = os.path.join(output_path,  kaind, 'images')
    os.makedirs(folder_path_img,  exist_ok=True)
    shutil.move(item,  os.path.join(folder_path_img, name))
    item_txt = item.replace('images', 'labels').replace('.jpg', '.txt')
    folder_path_lbl = folder_path_img.replace('images', 'labels')
    name_txt = name.replace('.jpg', '.txt')
    os.makedirs(os.path.join(output_path,  kaind, 'labels'),  exist_ok=True)
    shutil.move(item_txt, os.path.join(folder_path_lbl, name_txt))
    
    
if __name__ == "__main__":
    
    anno_path = "/home/skorp321/Projects/engineering/data/anno"
    output_path = "data/interim"
    anno_dict = {0: "tower crane", 1 :"hook"}
    
    procesed_files(anno_path, output_path, anno_dict)
    prepare_files(output_path, anno_dict)
    subprocess.run(['python3', '/home/skorp321/Projects/engineering/src/models/train_model.py'])