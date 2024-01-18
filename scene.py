# Selfie app V0.3
# Current funcionality: take photos & determine where the user's face is
# Created by: Tyler Cady
# Last Edit: 1.18.24
import cv2
# using yolov8 to detect objects in the image
import pygame


class SceneApp:
    def __init__(self):
        # Init camera, face detection, and sound
        self.capture = cv2.VideoCapture(0)
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=0.2)
        pygame.mixer.init()

    def take_photo(self):
        # Capturen a frame, save photo, play shutter sound
        ret, frame = self.capture.read()
        if ret:
            cv2.imwrite("photo.jpg", frame)
            print("Photo taken and saved as photo.jpg")
            self.play_sound("resources/shutter.mp3")

    def take_photo(self):
        # Capture a frame, save photo, play shutter sound
        ret, frame = self.capture.read()
        if ret:
            # Name photo with timestamp and add file extension
            timestamp = time.strftime("%y%m%d%H%M%S")
            file_name = f"selfie_{timestamp}.jpg"
            cv2.imwrite(file_name, frame)
            print(f"Photo taken and saved as {file_name}")
            self.play_sound("resources/shutter.mp3")
if __name__ == "__main__":
    app = SceneApp()
    app.run()
