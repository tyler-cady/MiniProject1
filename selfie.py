import cv2
import pygame

class SelfieApp:
    def __init__(self):
        self.capture = cv2.VideoCapture(0)
        pygame.mixer.init()

    def take_photo(self):
        ret, frame = self.capture.read()
        if ret:
            # Save the photo without the grid
            cv2.imwrite("photo.jpg", frame)
            print("Photo taken and saved as photo.jpg")

            # Sound Effect
            self.play_sound("shutter.mp3")

    def draw_grid(self, frame):
        # Draw the 2x2 grid and the fifth square in the center
        h, w, _ = frame.shape
        thickness = 2

        # Create a transparent overlay
        overlay = frame.copy()

        # Draw the main 2x2 grid on the overlay
        cv2.line(overlay, (w // 2, 0), (w // 2, h), (255, 255, 255), thickness)
        cv2.line(overlay, (0, h // 2), (w, h // 2), (255, 255, 255), thickness)

        # Draw the fifth square in the center on the overlay
        cv2.rectangle(overlay, (w // 4, h // 4), (3 * w // 4, 3 * h // 4), (255, 255, 255), thickness)

        # Blend the overlay with the original frame
        alpha = 0.5  # Adjust the alpha value for transparency
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    def play_sound(self, file_path):
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()

    def run(self):
        while True:
            ret, frame = self.capture.read()

            # Draw the transparent grid on the frame
            self.draw_grid(frame)

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
