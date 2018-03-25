import cv2
import os
import numpy as np
from .random_warp import random_warp

def read_images_from(path):
    valid_images = [".jpg",".gif",".png"]
    images = []
    wrap_images = []
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue

        image = cv2.imread(os.path.join(path,f))
        # images.append(image)
        wrap, target = random_warp(image)
        images.append(target)
        wrap_images.append(wrap)

    return np.array(wrap_images), np.array(images)
