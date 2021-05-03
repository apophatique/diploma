import cv2
import face_recognition
import numpy as np
from IO import IO
from image_transformations import ImageTransforms
from image_capture import ImageCapture
from work_states import SystemWorkStates


# TODO:
#   1. ImageTransforms: обрезать сразу np.array
#   2. (+) Database: сразу добавлять feature лица
#   3. (+) work_on_image_advanced(): исправить сравнение лиц через distance + np.argmin()
#   4. (*) Удалить из cv2 из контроллера


class RecognitionSystem:
    def __init__(self):
        self.methods_map = {
            SystemWorkStates.image: self.work_on_image,
            SystemWorkStates.video: self.work_on_video,
            SystemWorkStates.webcam: self.work_on_webcam
        }
        self.imagecapture = ImageCapture()

    def work_on_image(self):
        frame = self.imagecapture.capture_photo()
        known_face_encodings = [
            item.get_encoding() for item in self.database
        ]
        known_face_names = [
            item.get_name() for item in self.database
        ]
        face_names = []
        face_dists = []

        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for face_encoding in face_encodings:
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            name = known_face_names[best_match_index]
            face_names.append(name)
            dist = face_distances[best_match_index]
            face_dists.append(dist)
        return face_names, face_dists, frame, face_locations

    def work_on_video(self):
        video_fragment = self.imagecapture.capture_video()
        known_face_encodings = [
            item.get_encoding() for item in self.database
        ]
        known_face_names = [
            item.get_name() for item in self.database
        ]
        video_fragment = [
            fragment for index, fragment in enumerate(video_fragment) if index % 4 == 0
        ]
        face_names = set()
        face_dists = []
        for frame in video_fragment:
            small_frame = ImageTransforms.image_to_small(frame)
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            for face_encoding in face_encodings:
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                name = known_face_names[best_match_index]
                if name in face_names:
                    dist = face_distances[best_match_index]
                    face_dists[best_match_index] = (face_dists[best_match_index] + dist) / 2
                face_names.add(name)
            IO.show_video_frame(frame, face_locations, face_names)
            # Display the resulting image
            cv2.imshow('Video', frame)
            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # Release handle to the webcam
        cv2.destroyAllWindows()
        return face_names, face_dists, None, None

    def work_on_webcam(self):
        video_capture = cv2.VideoCapture(0)
        known_face_encodings = [
            item.get_encoding() for item in self.database
        ]
        known_face_names = [
            item.get_name() for item in self.database
        ]
        process_this_frame = True

        face_names = set()
        face_dists = {}
        while True:
            # Grab a single frame of video
            ret, frame = video_capture.read()
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Only process every other frame of video to save time
            if process_this_frame:
                # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                # Сравнение каждого детектированного лица с массивом лиц базы данных

                for face_encoding in face_encodings:
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)

                    name = known_face_names[best_match_index]
                    dist = face_distances[best_match_index]
                    if name not in face_names:
                        face_dists[name] = dist
                    else:
                        dist_pre = face_dists[name]
                        face_dists[name] = (dist_pre + dist) / 2
                    face_names.add(name)

            process_this_frame = not process_this_frame
            # Display the results
            IO.show_video_frame(frame, face_locations, face_names)
            # Display the resulting image
            cv2.imshow('Video', frame)
            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # Release handle to the webcam
        video_capture.release()
        cv2.destroyAllWindows()
        return face_names, face_dists, frame, face_locations

    def run(self, mode, database):
        self.database = database
        work_method = self.methods_map.get(mode)
        results, frame, boxes = work_method()
        return results, frame, boxes
