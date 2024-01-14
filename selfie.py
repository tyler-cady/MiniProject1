import cv2
import pygame
import os

class SelfieApp:
    def __init__(self):
        self.capture = cv2.VideoCapture(0)
        pygame.mixer.init()

    def take_photo(self):
        ret, frame = self.capture.read()
        if ret:
            cv2.imwrite("photo.jpg", frame)
            print("Photo taken and saved as photo.jpg")
            # Sound Effect
            self.play_sound("shutter.mp3")

    def play_sound(self, file_path):
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()

    def run(self):
        while True:
            ret, frame = self.capture.read()
            cv2.imshow("Selfie App", frame)

            key = cv2.waitKey(1)
            if key == ord('q'):  # Press 'q' to quit
                break
            elif key == ord('s'):  # Press 's' to take a photo
                self.take_photo()

        self.capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = SelfieApp()
    app.run()
