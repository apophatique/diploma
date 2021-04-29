import os
import cv2
from student import Student

class Database:
    def __init__(self):
        pass

    def get_database(self, path):
        student_database = []
        for filename in os.listdir(path):
            new_student = Student(filename.split('.')[0],
                                  cv2.imread(path + '/' + filename, cv2.IMREAD_COLOR))
            student_database.append(new_student)

        return student_database