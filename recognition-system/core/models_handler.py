import sys
import logging.config
import yaml
import numpy as np
from PIL import Image

from FaceDetModelLoader import FaceDetModelLoader
from FaceDetModelHandler import FaceDetModelHandler
from FaceAlignModelLoader import FaceAlignModelLoader
from FaceAlignModelHandler import FaceAlignModelHandler
from FaceRecModelLoader import FaceRecModelLoader
from FaceRecModelHandler import FaceRecModelHandler
from FaceRecImageCropper import FaceRecImageCropper

logging.config.fileConfig("../config/logging.conf")
logger = logging.getLogger('api')

with open('../config/model_conf.yaml') as f:
    model_conf = yaml.load(f)


class ModelsHandler:
    def __init__(self):
        self.model_path = '../models'
        self.scene = 'non-mask'
        self.load_models()
        self.face_cropper = FaceRecImageCropper()

    def load_models(self):
        logger.info('Start to load the face recognition model...')
        try:
            model_category = 'face_recognition'
            model_name = model_conf[self.scene][model_category]

            faceRecModelLoader = FaceRecModelLoader(self.model_path, model_category, model_name)
            model, cfg = faceRecModelLoader.load_model()
            self.faceRecModelHandler = FaceRecModelHandler(model, 'cpu', cfg)
        except Exception as e:
            logger.error('Model loading failed!')
            logger.error(e)
            sys.exit(-1)
        else:
            logger.info('Successfully loaded the face recognition model!')

        logger.info('Start to load the face landmark model...')
        try:
            model_category = 'face_alignment'
            model_name = model_conf[self.scene][model_category]
            faceAlignModelLoader = FaceAlignModelLoader(self.model_path, model_category, model_name)
            model, cfg = faceAlignModelLoader.load_model()
            self.faceAlignModelHandler = FaceAlignModelHandler(model, 'cpu', cfg)
        except Exception as e:
            logger.error('Failed to load face landmark model.')
            logger.error(e)
            sys.exit(-1)
        else:
            logger.info('Success!')

        logger.info('Start to load the face detection model...')
        try:
            model_category = 'face_detection'
            model_name = model_conf[self.scene][model_category]

            faceDetModelLoader = FaceDetModelLoader(self.model_path, model_category, model_name)
            model, cfg = faceDetModelLoader.load_model()
            self.faceDetModelHandler = FaceDetModelHandler(model, 'cpu', cfg)
        except Exception as e:
            logger.error('Model loading failed!')
            logger.error(e)
            sys.exit(-1)
        else:
            logger.info('Successfully loaded the face detection model!')

    def getRecognitionModel(self):
        return self.faceRecModelHandler

    def get_landmark_model(self):
        return self.faceAlignModelHandler

    def get_detection_model(self):
        return self.faceDetModelHandler

    def get_image_feature(self, image: Image, bbox=None, full_image=False):
        if bbox is None:
            width, height, _ = image.shape
            bbox = [0, 0, width, height]

        landmarks = self.faceAlignModelHandler.inference_on_image(image, bbox)
        landmarks_list = []
        for (x, y) in landmarks.astype(np.int32):
            landmarks_list.extend((x, y))
        detected_cropped_face = self.face_cropper.crop_image_by_mat(image, landmarks_list)
        detected_feature = self.faceRecModelHandler.inference_on_image(detected_cropped_face)
        return detected_feature
