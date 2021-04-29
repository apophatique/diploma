import cv2
import time


class ImageCapture:
    def __init__(self):
        pass

    def capture_photo(self, save=False):
        cap = cv2.VideoCapture(0)
        cap.set(3, 1280)
        cap.set(4, 720)
        read_status, frame = cap.read()
        if read_status:
            cv2.imshow('face Capture', frame)

        cap.release()
        cv2.destroyAllWindows()
        if save:
            cv2.imwrite('zxc.jpg', frame)
        return frame

    def capture_video(self, recording_time=3):
        frames_list = []

        cap = cv2.VideoCapture(0)
        cap.set(3, 1280)
        cap.set(4, 720)

        start_time = time.time()
        while time.time() - start_time < recording_time:
            read_status, frame = cap.read()
            if read_status:
                frames_list.append(frame)

            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        return frames_list



if __name__ == '__main__':
    capture = ImageCapture()
    # capture.capture_photo()
    frames_list = capture.capture_video()
    frame = frames_list[len(frames_list) - 1]
    print(len(frames_list))