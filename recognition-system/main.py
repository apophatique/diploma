import cv2
import numpy as np
from PIL import Image

import os
import sys
import logging.config

from student import Student
from FaceRecImageCropper import FaceRecImageCropper

sys.path.append('..')
logging.config.fileConfig("../config/logging.conf")
logger = logging.getLogger('api')

config = {
    'input_path': 'input/img_1.png',
    'recognition_threshold': 0.1,
    'save_image': True,
    'save_txt': False,
    'output_path': 'output/test1_detect_res.jpg',
    'save_path_txt': '../output/test1_detect_res.txt'
}

from core import RecognitionSystem

if __name__ == '__main__':
    recognition_system = RecognitionSystem()
    recognition_system.start()

def zxc():
    image_path = config['input_path']
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    face_cropper = FaceRecImageCropper()

    try:
        bboxs = detectionModel.inference_on_image(image)
    except Exception as e:
        logger.error('Face detection failed!')
        logger.error(e)
        sys.exit(-1)

    face_area_list = [
        (bbox[0],
         bbox[1],
         bbox[2],
         bbox[3]) for bbox in bboxs
    ]
    student_database = []
    for filename in os.listdir('../database'):
        new_student = Student(filename.split('.')[0],
                              cv2.imread('database/' + filename, cv2.IMREAD_COLOR))
        student_database.append(new_student)

    results = {}
    for i, bbox in enumerate(face_area_list):
        landmarks = alignmentModel.inference_on_image(image, bbox)
        landmarks_list = []
        for (x, y) in landmarks.astype(np.int32):
            landmarks_list.extend((x, y))
        detected_cropped_face = face_cropper.crop_image_by_mat(image, landmarks_list)
        detected_feature = recognitionModel.inference_on_image(detected_cropped_face)

        for j, student in enumerate(student_database):
            db_face = student.get_face()
            width, height, _ = db_face.shape
            tmp_bbox = [0, 0, width, height]
            landmarks = alignmentModel.inference_on_image(db_face, tmp_bbox)

            landmarks_list = []
            for (x, y) in landmarks.astype(np.int32):
                landmarks_list.extend((x, y))
            db_cropped_face = face_cropper.crop_image_by_mat(db_face, landmarks_list)
            db_feature = recognitionModel.inference_on_image(db_cropped_face)

            score = np.dot(db_feature, detected_feature)

            if (score > config['recognition_threshold']):
                if (j not in results) or (j in results and score > results[j]['score']):
                    results[j] = {
                        'name': student.get_name(),
                        'bbox': bbox,
                        'score': score
                    }

    for key in results:
        item = results[key]
        face = image_pil.crop(item['bbox'])
        face.save('output/faces_detected/' + str(item['name']) + '-' + str(item['score']) + '.png')

    def get_feature():
        pass
