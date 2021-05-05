class Student:
    def __init__(self, name, face_image, face_encoding):
        self.name = name
        self.face_image = face_image
        self.face_encoding = face_encoding

    def get_face(self):
        return self.face_image

    def get_name(self):
        return self.name

    def get_encoding(self):
        return self.face_encoding

    def __str__(self):
        return self.name
