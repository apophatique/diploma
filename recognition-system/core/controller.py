import face_recognition
import numpy as np
from image_transformations import ImageTransforms
from work_states import SystemWorkStates
from image_capture import ImageCapture
from recognizer import StudentRecognizer
from database.student_database import Database
from models_handler import ModelsHandler
from IO import IO
import cv2

config = {
    'recognition_threshold': 0.1,
    'save_frame': True,
    'save_results': True,
    'frame_output_path': '../output/frame.jpg',
    'results_output_path': '../output/test1_detect_res.txt'
}

# TODO:
#   1. ImageTransforms: обрезать сразу np.array
#   2. (+) Database: сразу добавлять feature лица
#   3. work_on_image_advanced(): исправить сравнение лиц через distance + np.argmin()
#   4. Удалить из cv2 из контроллера


class RecognitionSystem:
    def __init__(self):
        self.database_path = '../input/database/popular/'
        self.methods_map = {
            SystemWorkStates.image: self.work_on_image,
            SystemWorkStates.video: self.work_on_video,
            SystemWorkStates.webcam: self.work_on_webcam,
            SystemWorkStates.image_advanced: self.work_on_image_advanced
        }
        self.imagecapture = ImageCapture()
        self.recognizer = StudentRecognizer()
        self.database = Database()
        self.model_handler = ModelsHandler()

    def work_on_image(self):
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
        for i, bbox in enumerate(face_area_list):
            detected_feature = self.model_handler.get_image_feature(frame, bbox)
            for j, student in enumerate(student_database):
                db_face = student.get_face()
                db_feature = self.model_handler.get_image_feature(db_face)
                score = np.dot(db_feature, detected_feature)
                if score > config['recognition_threshold']:
                    if (j not in results) or (j in results and score > results[j]['score']):
                        results[j] = {
                            'name': student.get_name(),
                            'bbox': bbox,
                            'score': score
                        }
                print(score)
        IO.save_image_with_detections(frame, bboxs, config['frame_output_path'])
        IO.save_detected_faces(frame, results)

    def work_on_image_advanced(self):
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
        for i, bbox in enumerate(face_area_list):
            detected_face = ImageTransforms.preprocess_pil(frame, bbox)
            detected_feature = face_recognition.face_encodings(detected_face)[0]

            for j, student in enumerate(student_database):
                db_face = student.get_face()
                db_feature = face_recognition.face_encodings(db_face)[0]

                is_one_person = face_recognition.compare_faces([detected_feature], db_feature)
                if is_one_person[0] and j not in results:
                    results[j] = {
                        'name': student.get_name(),
                        'bbox': bbox
                    }
        IO.save_image_with_detections(frame, bboxs, config['frame_output_path'])
        IO.save_detected_faces(frame, results)

    def work_on_video(self):
        video_fragment = self.imagecapture.capture_video()
        student_database = self.database.get_database(self.database_path)
        known_face_encodings = [
            item.get_encoding() for item in student_database
        ]
        known_face_names = [
            item.get_name() for item in student_database
        ]

        for frame in video_fragment:
            small_frame = ImageTransforms.image_to_small(frame)
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

                best_match_index = np.argmin(face_distances)
                print(best_match_index)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)

            IO.show_video_frame(frame, face_locations, face_names)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


    def work_on_webcam(self):
        video_capture = cv2.VideoCapture(0)
        student_database = self.database.get_database(self.database_path)
        known_face_encodings = [
            item.get_encoding() for item in student_database
        ]
        known_face_names = [
            item.get_name() for item in student_database
        ]
        process_this_frame = True
        while True:
            # Grab a single frame of video
            ret, frame = video_capture.read()

            # Resize frame of video to 1/4 size for faster face recognition processing

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = frame[:, :, ::-1]

            # Only process every other frame of video to save time
            if process_this_frame:
                # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                face_names = []
                for face_encoding in face_encodings:
                    # See if the face is a match for the known face(s)
                    name = "Unknown"

                    # # If a match was found in known_face_encodings, just use the first one.
                    # if True in matches:
                    #     first_match_index = matches.index(True)
                    #     name = known_face_names[first_match_index]

                    # Or instead, use the known face with the smallest distance to the new face
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

                    best_match_index = np.argmin(face_distances)
                    name = known_face_names[best_match_index]

                    face_names.append(name)

            process_this_frame = not process_this_frame

            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                '''  
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                '''
                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            # Display the resulting image
            cv2.imshow('Video', frame)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release handle to the webcam
        video_capture.release()
        cv2.destroyAllWindows()

    def start(self, mode=SystemWorkStates.webcam):
        work_method = self.methods_map.get(mode)
        work_method()
