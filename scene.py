# Static Scene App V 0.1
# Current funcionality: 
# Created by: Tyler Cady, 
# Last Edit: 1.18.24
import cv2
# using yolov8 to detect objects in the image
import pygame
import time
from ultralytics import YOLO

# Load the model
model = YOLO('Yolov8n.pt')
class SceneApp:

    def __init__(self):
        # Init camera, face detection, and sound
        self.capture = cv2.VideoCapture(0)
        pygame.mixer.init()

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
    
    def play_sound(self, file_path):
        """
        Play a sound from a file.
        """
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()


if__name__ == "__main__":
    app = SceneApp()
    app.run()
