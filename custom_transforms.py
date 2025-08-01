# custom_transforms.py
import cv2
import numpy as np
from PIL import Image

class HistogramEqualization:
    def __call__(self, img):
        img = np.array(img)
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
            img_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        else:
            img_eq = cv2.equalizeHist(img)
        return Image.fromarray(img_eq)

class Denoise:
    def __call__(self, img):
        img = np.array(img)
        img_blur = cv2.GaussianBlur(img, (3, 3), 0)
        return Image.fromarray(img_blur)
