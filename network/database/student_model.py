import numpy as np

class Student:
    def __init__(self, data):
        self.name = data[0]
        self.surname = data[1]
        self.patronymic = data[2]
        self.sex = data[3]
        self.group = data[4]
        self.face_image = data[5]
        self.face_encoding = np.fromstring(data[6], dtype=np.float64)

    def get_face(self):
        return self.face_image

    def get_name(self):
        return self.name

    def get_encoding(self):
        return self.face_encoding

    def __str__(self):
        return self.name
