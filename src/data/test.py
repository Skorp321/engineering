import os

data_path = 'video'

for root, dirs, files in os.walk(data_path):
    print(f'root: {root}, dirs: {dirs}')
    for file in files:
        print(file)