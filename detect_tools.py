from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import numpy as np
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8x.pt')  # load an official model

# from ndarray
im2 = cv2.imread("/mnt/L3MVN/tmp/dump/llava_nav/episodes/thread_1/eps_7/1-7-Obs-90.png")
results = model.predict(source=im2, save=True, save_txt=True, conf=0.7, device="cuda:5", augment=True)  # save predictions as labels


# Iterate over detection results
r = results[0]
# Iterate over each object contour
for ci, c in enumerate(r):
    label = c.names[c.boxes.cls.tolist().pop()]
    print(label)
    