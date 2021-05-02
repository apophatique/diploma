import cv2
import numpy as np
from PIL import Image


class ImageTransforms:
    @staticmethod
    def preprocess_pil(image, bbox):
        image_pil = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_pil)
        image_pil = image_pil.crop(bbox)
        im = image_pil.convert('RGB')
        return np.array(im)

    @staticmethod
    def pil_to_cv2(image):
        mode = 'RGB'
        image = image.convert(mode)
        return np.array(image)

    @staticmethod
    def image_to_small(image):
        return cv2.resize(image, (0, 0), fx=0.25, fy=0.25)

