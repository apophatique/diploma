import sqlite3 as sl
import os
import cv2
import face_recognition

from student_model import Student


class Database:
    def __init__(self, path):
        self.path = path
        self.create_db()

    def create_db(self):
        with sl.connect('students.db') as con:
            self.cur = con.cursor()
            drop_table_query = '''
                DROP TABLE IF EXISTS students
            '''
            self.cur.execute(drop_table_query)

            create_db_query = '''
                CREATE TABLE IF NOT EXISTS students (
                    name TEXT NOT NULL,
                    surname TEXT NOT NULL,
                    patronymic TEXT,
                    sex TEXT NOT NULL,
                    [group] TEXT NOT NULL,
                    face_image BLOB NOT NULL,
                    face_encoding TEXT NOT NULL
                )
            '''
            self.cur.execute(create_db_query)

            insert_query = '''
                INSERT INTO students VALUES (?, ?, ?, ?, ?, ?, ?)
            '''
            for filename in os.listdir(self.path):
                img_path = self.path + filename
                filename = filename.split('.')[0]
                name, surname, patronymic, sex, group = filename.split('_')[0], \
                                                        filename.split('_')[1], \
                                                        filename.split('_')[2], \
                                                        filename.split('_')[3], \
                                                        filename.split('_')[4]
                pict_binary = self.import_pict_binary(img_path)
                print(img_path)
                face_encoding = self.get_face_encoding(img_path)
                face_encoding = sl.Binary(face_encoding)
                data = (name, surname, patronymic, sex, group, pict_binary, face_encoding)
                self.cur.execute(insert_query, data)

    def import_pict_binary(self, path):
        f = open(path, 'rb')
        pict_binary = f.read()
        return pict_binary

    def get_face_encoding(self, path):
        face = cv2.imread(path)
        return face_recognition.face_encodings(face)[0]

    def get_students_data(self, group):
        select_query = '''
            SELECT * FROM students WHERE [group] in (?)
        '''
        if type(group) is list:
            print('zxc: ', (group))
            self.cur.execute(select_query, group)
        else:
            self.cur.execute(select_query, (group,))
        items = self.cur.fetchall()

        students = []
        for item in items:
            students.append(Student(item))
        return students

    def drop_row(self, group):
        query = '''
            DELETE FROM students WHERE [group] in ?
        '''
        self.cur.execute(query, (group, ))


if __name__ == '__main__':
    group = ['PIN-171', 'FIT-211']
    database = Database('D:\Projects\diploma/input/database/')

