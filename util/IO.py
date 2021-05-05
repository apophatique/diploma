import cv2
from PIL import Image
import logging.config
import csv
import numpy as np

class IO:
    @staticmethod
    def save_image_with_detections(image, bboxs, output_path):
        for box in bboxs:
            box = list(map(int, box))
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 255, 255), 2)
        cv2.imwrite(output_path, image)

    @staticmethod
    def write_bboxes(bboxs, path):
        with open(path, "w") as fd:
            for box in bboxs:
                line = str(int(box[0])) + " " + str(int(box[1])) + " " + \
                       str(int(box[2])) + " " + str(int(box[3])) + " " + \
                       str(box[4]) + " \n"
                fd.write(line)

    @staticmethod
    def save_detected_faces(results, frame, path):
        for name in results:
            item = results[name]
            (top, right, bottom, left) = item['bbox']
            bbox = [
                left,
                top,
                right,
                bottom
            ]
            image_pil = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_pil)
            face = image_pil.crop(bbox)
            face.save(path + name + '.png')

    @staticmethod
    def save_image(image, output_path):
        cv2.imwrite(output_path, image)

    @staticmethod
    def show_video_frame(results, frame):
        # Display the results
        for name in results:
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            item = results[name]
            (top, right, bottom, left) = item['bbox']
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)

    @staticmethod
    def write_csv_results(results, path):
        with open(path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(('Student', 'Confidence'))
            for name in results:
                writer.writerow((name, results[name]['confidence']))
