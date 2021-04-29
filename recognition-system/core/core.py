import numpy as np
import face_recognition
import cv2

from enums import SystemModeStates
from imagecapture import ImageCapture
from recognizer import StudentRecognizer
from database.student_database import Database
from models_handler import ModelsHandler
from IOController import IOController

config = {
    'recognition_threshold': 0.1,
    'save_frame': True,
    'save_results': True,
    'frame_output_path': '../output/frame.jpg',
    'results_output_path': '../output/test1_detect_res.txt'
}

class RecognitionSystem:
    database_path = '../database'

    def __init__(self):
        self.imagecapture = ImageCapture()
        self.recognizer = StudentRecognizer()
        self.database = Database()
        self.model_handler = ModelsHandler()
        self.io = IOController()

    def start(self,
              mode=SystemModeStates.post,
              recording_time=3):
        # zxcqwe123deadinside14y.o.1posorfeed
        frame = self.imagecapture.capture_photo()
        student_database = self.database.get_database(self.database_path)

        bboxs = self.model_handler.get_detection_model().inference_on_image(frame)
        face_area_list = [
            (bbox[0],
             bbox[1],
             bbox[2],
             bbox[3]) for bbox in bboxs
        ]

        results = {}
        print('ZXCQWE12R21R1F', len(student_database))
        for i, bbox in enumerate(face_area_list):
            detected_feature = self.model_handler.get_image_feature(frame, bbox)

            print(len(student_database))
            for j, student in enumerate(student_database):
                db_face = student.get_face()
                db_feature = self.model_handler.get_image_feature(db_face)

                score = np.dot(db_feature, detected_feature)
                if (score > config['recognition_threshold']):
                    if (j not in results) or (j in results and score > results[j]['score']):
                        results[j] = {
                            'name': student.get_name(),
                            'bbox': bbox,
                            'score': score
                        }

        self.io.save_image_with_detections(frame, bboxs, config['frame_output_path'])
        self.io.save_detected_faces(frame, results)
        cv2.imwrite(config['frame_output_path'], frame)
