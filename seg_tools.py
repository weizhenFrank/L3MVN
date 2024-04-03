from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import numpy as np
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8x-seg.pt')  # load an official model

# from ndarray
im2 = cv2.imread("/mnt/L3MVN/tmp/dump/llava_nav/episodes/thread_1/eps_7/1-7-Obs-59.png")
results = model.predict(source=im2, save=True, save_txt=True, conf=0.5, device="cuda:0", augment=True)  # save predictions as labels


# Iterate over detection results
r = results[0]
print(r.plot().shape)
img = np.copy(r.orig_img)
img_name = Path(r.path).stem


# Iterate over each object contour
for ci, c in enumerate(r):
    label = c.names[c.boxes.cls.tolist().pop()]
    
    class_masks = np.zeros(img.shape[:2], dtype=np.uint8)
    bin_masks = np.zeros(img.shape[:2], dtype=np.uint8)
    
    # Create contour mask
    contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
    _ = cv2.drawContours(class_masks, [contour], -1, 255, cv2.FILLED)
    _ = cv2.drawContours(bin_masks, [contour], -1, 1, cv2.FILLED)
    print(f"Class: {label}")
    print((class_masks == 255).sum())
    print((bin_masks == 1).sum())
    cv2.imwrite(f"{ci}_mask.png", class_masks)
 