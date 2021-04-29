class Student:
    def __init__(self, name, face_image):
        self.name = name
        self.face_image = face_image

    def get_face(self):
        return self.face_image

    def get_name(self):
        return self.name

    def __str__(self):
        return self.name