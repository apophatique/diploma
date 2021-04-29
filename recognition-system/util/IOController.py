import cv2
import logging.config

class IOController:
    def __init__(self):
        pass

    def save_image_with_detections(self, image, bboxs, output_path):
        for box in bboxs:
            box = list(map(int, box))
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 0), 2)
        cv2.imwrite(output_path, image)

    def write_bboxes(self, bboxs, path):
        with open(path, "w") as fd:
            for box in bboxs:
                line = str(int(box[0])) + " " + str(int(box[1])) + " " + \
                       str(int(box[2])) + " " + str(int(box[3])) + " " + \
                       str(box[4]) + " \n"
                fd.write(line)

    def save_detected_faces(self, image, results):
        for key in results:
            item = results[key]
            face = image.crop(item['bbox'])
            face.save('output/faces_detected/' + str(item['name']) + '-' + str(item['score']) + '.png')

