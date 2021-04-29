import cv2
import insightface
import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms

import logging.config
import os
import sys

from student import Student
from core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader
from core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler

sys.path.append('..')
logging.config.fileConfig("../config/logging.conf")
logger = logging.getLogger('api')

with open('../config/model_conf.yaml') as f:
    model_conf = yaml.load(f)

config = {
    'recognition_threshold': 0.5
}

embedder = insightface.iresnet100()
embedder.eval()

mean = [0.5] * 3
std = [0.5 * 256 / 255] * 3
preprocess = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

if __name__ == '__main__':
    model_path = '../models'
    scene = 'non-mask'

    logger.info('Start to load the face detection model...')
    try:
        model_category = 'face_detection'
        model_name = model_conf[scene][model_category]

        faceDetModelLoader = FaceDetModelLoader(model_path, model_category, model_name)
        model, cfg = faceDetModelLoader.load_model()
        faceDetModelHandler = FaceDetModelHandler(model, 'cpu', cfg)
    except Exception as e:
        logger.error('Model loading failed!')
        logger.error(e)
        sys.exit(-1)
    else:
        logger.info('Successfully loaded the face detection model!')
    # read image
    image_path = '../input/img_1.png'
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    try:
        bboxs = faceDetModelHandler.inference_on_image(image)
    except Exception as e:
        logger.error('Face detection failed!')
        logger.error(e)
        sys.exit(-1)
    else:
        logger.info('Successful face detection!')

    # gen result
    save_path_img = '../output/test1_detect_res.jpg'
    save_path_txt = 'test1_detect_res.txt'

    with open(save_path_txt, "w") as fd:
        for box in bboxs:
            line = str(int(box[0])) + " " + str(int(box[1])) + " " + \
                   str(int(box[2])) + " " + str(int(box[3])) + " " + \
                   str(box[4]) + " \n"
            fd.write(line)

    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    face_area_list = [
        (bbox[0],
         bbox[1],
         bbox[2],
         bbox[3]) for bbox in bboxs
    ]

    for box in bboxs:
        box = list(map(int, box))
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
    cv2.imwrite(save_path_img, image)
    logger.info('Successfully generate face detection results!')

    student_database = []
    for filename in os.listdir('../database'):
        new_student = Student(filename,
                              Image.open('database/' + filename))
        student_database.append(new_student)

    for i, student in enumerate(student_database):
        db_face = student.get_face()
        for j, bbox in enumerate(face_area_list):
            detected_face = image_pil.crop(bbox)

            tensor1 = preprocess(detected_face)
            tensor2 = preprocess(db_face)

            with torch.no_grad():
                features1 = embedder(tensor1.unsqueeze(0))[0]
                features2 = embedder(tensor2.unsqueeze(0))[0]

                feature1 = features1.cpu().numpy()
                feature2 = features2.cpu().numpy()

                x1 = feature1 / np.linalg.norm(feature1)
                x2 = feature2 / np.linalg.norm(feature2)

                feature1 = np.expand_dims(feature1, axis=0)
                feature2 = np.expand_dims(feature2, axis=0)

            score = np.dot(x1, x2.T)
            logger.info(score)
            if (score > 0.3):
                detected_face.save('output/faces_detected/' + str(student.get_name()) + '_' + str(i) + '-' + str(j) + '.png')

    logger.info(len(student_database))