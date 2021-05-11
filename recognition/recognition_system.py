import cv2
import face_recognition
import numpy as np
from IO import IO
from image_transformations import ImageTransforms
from image_capture import ImageCapture
from work_states import SystemWorkStates


# TODO:
#   1. (-) ImageTransforms: обрезать сразу np.array
#   2. (+) Database: сразу добавлять feature лица
#   3. (+) work_on_image_advanced(): исправить сравнение лиц через distance + np.argmin()
#   4. (+) Удалить из cv2 из контроллера
#   5.

class RecognitionSystem:
    def __init__(self):
        self.methods_map = {
            SystemWorkStates.image.value: self.work_on_image,
            SystemWorkStates.video.value: self.work_on_video,
            SystemWorkStates.webcam.value: self.work_on_webcam
        }
        self.imagecapture = ImageCapture()

    def work_on_image(self):
        frame = self.imagecapture.capture_photo()
        known_face_encodings = [
            item.get_encoding() for item in self.data_students
        ]
        known_face_names = [
            item.get_name() for item in self.data_students
        ]
        results = {}
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            name = known_face_names[best_match_index]
            dist = face_distances[best_match_index]
            results[name] = {
                'confidence': dist,
                'bbox': face_location
            }
        return results, frame

    def work_on_webcam(self):
        video_capture = cv2.VideoCapture(0)
        known_face_encodings = [
            item.get_encoding() for item in self.data_students
        ]
        known_face_names = [
            item.get_name() for item in self.data_students
        ]
        results = {}
        tmp_result = {}
        process_this_frame = True

        while True:
            # Grab a single frame of video
            ret, frame = video_capture.read()
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Only process every other frame of video to save time
            if process_this_frame:
                tmp_result.clear()
                # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                # Сравнение каждого детектированного лица с массивом лиц базы данных

                for face_encoding, face_location in zip(face_encodings, face_locations):
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    name = known_face_names[best_match_index]
                    dist = face_distances[best_match_index]
                    tmp_result[name] = {
                        'confidence': dist,
                        'bbox': face_location
                    }
            # Display the results
            IO.show_video_frame(tmp_result, frame)
            for name in tmp_result:
                if name not in results.keys():
                    results[name] = {
                        'confidence': tmp_result[name]['confidence']
                    }
                else:
                    results[name]['confidence'] = (results[name]['confidence'] + tmp_result[name]['confidence']) / 2
            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            process_this_frame = not process_this_frame
        # Release handle to the webcam
        video_capture.release()
        cv2.destroyAllWindows()
        return results, frame

    def work_on_video(self):
        video_fragment = self.imagecapture.capture_video()
        known_face_encodings = [
            item.get_encoding() for item in self.data_students
        ]
        known_face_names = [
            item.get_name() for item in self.data_students
        ]
        video_fragment = [
            fragment for index, fragment in enumerate(video_fragment) if index % 4 == 0
        ]
        results = {}
        tmp_result = {}
        for frame in video_fragment:
            small_frame = ImageTransforms.image_to_small(frame)
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            for face_encoding, face_location in zip(face_encodings, face_locations):
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                name = known_face_names[best_match_index]
                dist = face_distances[best_match_index]
                tmp_result[name] = {
                    'confidence': dist,
                    'bbox': face_location
                }
            for face_encoding, face_location in zip(face_encodings, face_locations):
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                name = known_face_names[best_match_index]
                dist = face_distances[best_match_index]
                if name not in results:
                    results[name] = {
                        'confidence': dist
                    }
                else:
                    results[name]['confidence'] = (results[name]['confidence'] + dist) / 2
                results[name]['bbox'] = face_location
            IO.show_video_frame(tmp_result, frame)
            for name in tmp_result:
                if name not in results.keys():
                    results[name] = {
                        'confidence': tmp_result[name]['confidence']
                    }
                else:
                    results[name]['confidence'] = (results[name]['confidence'] + tmp_result[name]['confidence']) / 2
            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # Release handle to the webcam
        cv2.destroyAllWindows()
        return results, video_fragment

    def run(self, mode, data_students):
        self.data_students = data_students
        work_method = self.methods_map.get(mode)
        results, frame = work_method()
        return results, frame
