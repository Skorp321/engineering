import cv2
import pandas as pd

img_path = 'data\\interim\\Black_river\\images\\000001.jpg'
txt_path = 'data\\interim\\Black_river\\labels\\000001.txt'

txt_data = pd.read_table(txt_path, header=None, sep=" ")
img = cv2.imread(img_path)
height, weight, _ = img.shape
print(weight, height)
for _, item in txt_data.iterrows():
    cls, x, y, w, h = item
    x1 = int(x * weight)
    y1 = int(y * height)
    x2 = int(x1 + w * weight)
    y2 = int(y1 + h * height)
    
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.imshow('img', img)
cv2.waitKey(0)