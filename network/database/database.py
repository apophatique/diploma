import sqlite3 as sl
import os
import cv2
import face_recognition

from student_model import Student


class Database:
    def __init__(self, path):
        # self.create_db()
        self.path = path
        # cv2.imread(path + '/' + filename, cv2.IMREAD_COLOR)

    def create_db(self):
        self.con = sl.connect('students.db')
        cur = self.con.cursor()
        create_db_query = '''
            CREATE TABLE IF NOT EXISTS students (
                name TEXT NOT NULL,
                surname TEXT NOT NULL,
                patronymic TEXT,
                sex INTEGER NOT NULL,
                group INTEGER NOT NULL,
                photo BLOB NOT NULL
            )
        '''
        # student_id INTEGER PRIMARY KEY AUTOINCREMENT,

        self.con.execute(create_db_query)

        insert_query = '''
            INSERT INTO students VALUES (?, ?, ?, ?, ?, ?, ?)
        '''
        for filename in os.listdir(self.path):
            name, surname, patronymic, sex, group = filename.split('_')[0], \
                                                    filename.split('_')[1], \
                                                    filename.split('_')[2], \
                                                    filename.split('_')[3], \
                                                    filename.split('_')[4]
            pict_binary = self.import_pict_binary(self.path + '/' + filename)
            data = (name, surname, patronymic, sex, group, pict_binary)
            self.con.execute(insert_query, data)

    def import_pict_binary(self, path):
        f = open(path, 'rb')
        pict_binary = f.read()
        return pict_binary

    def get_students(self, group):
        select_query = '''
            SELECT * FROM students WHERE group == ?
        '''
        data = self.con.execute(select_query, group)
        return data

    def get_database_from_folder(self):
        student_database = []

        for filename in os.listdir(self.path):
            face_image = cv2.imread(self.path + filename, cv2.IMREAD_COLOR)
            new_student = Student(
                filename.split('.')[0],  # Имя студента
                face_image,  # Фото лица формата np.array
                face_recognition.face_encodings(face_image)[0]  # Вектор, полученный из фото лица
            )
            student_database.append(new_student)
        return student_database


if __name__ == '__main__':
    database = Database('input/database/')
