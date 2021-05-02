import os
import cv2
import face_recognition

from student import Student


class Database:
    def __init__(self):
        pass

    def get_database(self, path):
        student_database = []

        for filename in os.listdir(path):
            face_image = cv2.imread(path + '/' + filename, cv2.IMREAD_COLOR)
            new_student = Student(
                filename.split('.')[0],
                face_image,
                face_recognition.face_encodings(face_image)[0]
            )
            student_database.append(new_student)

        return student_database
