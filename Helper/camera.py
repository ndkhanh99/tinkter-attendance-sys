import cv2, threading

class VideoCamera(object):

    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.is_record = False
        self.out = None
        self.recordingThread = None

    def __del__(self):
        self.cap.release()

    def get_frame(self):
        ret, frame = self.cap.read()
        if ret:
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()
        return

    def stop_stream(self):
        self.cap.release()
        cv2.destroyAllWindows()
        return print('close success!')