import cv2
import pygame
import mediapipe as mp
import random
import os

class SelfieApp:

    def __init__(self):
        # Init camera, face detection, and sound
        self.capture = cv2.VideoCapture(0)
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.2)
        pygame.mixer.init()

        # Set up Pygame
        pygame.init()
        self.WIDTH, self.HEIGHT = 800, 600
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Selfie App Tester")

        # Checklist items and order
        self.checklist_items = ["Top Right", "Top Left", "Bottom Right", "Bottom Left", "Center"]
        self.randomized_checklist = random.sample(self.checklist_items, len(self.checklist_items))

        # Crossed off items
        self.crossed_off_items = set()

        # Subject's name input
        self.subject_name = ""
        self.input_rect = pygame.Rect(self.WIDTH // 2 - 100, 60, 200, 30)
        self.input_active = False
        self.input_font = pygame.font.Font(None, 32)
        self.input_text = ""

        # Photo count for each participant
        self.photo_count = 0

    def take_photo(self):
        # Capture a frame, save photo with participant's name and a number, play shutter sound, cross off an item
        ret, frame = self.capture.read()
        if ret:
            # Generate file name with the first letter of the participant's name and a number
            file_name = f"{self.subject_name[:1].upper()}_photo_{self.photo_count}.jpg"
            cv2.imwrite(file_name, frame)
            print(f"Photo taken and saved as {file_name}")
            self.play_sound("shutter.mp3")
            self.photo_count += 1
            self.cross_off_item()

    def play_sound(self, file_path):
        # multi-purpose sound-playing function
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()

    def determine_quad_pct(self, frame):
        # Determine the percentage of the user's face that lies in each of the 4 quadrants
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        if results.detections:
            detection = results.detections[0]
            bboxC = detection.location_data.relative_bounding_box
            x, y, width, height = tuple(map(int, (bboxC.xmin * w, bboxC.ymin * h, bboxC.width * w, bboxC.height * h)))
            total_face_area = width * height
            square1_area = ((w // 2) - x) * ((h // 2) - y)
            square2_area = (x + width - (w // 2)) * ((h // 2) - y)
            square3_area = ((w // 2) - x) * (y + height - (h // 2))
            square4_area = (x + width - (w // 2)) * (y + height - (h // 2))
            pct_tuple = tuple(((square_area / total_face_area) * 100)
                              for square_area in (square1_area, square2_area, square3_area, square4_area))
            return pct_tuple
        else:
            return (0, 0, 0, 0)

    def print_quadrant(self, pct_tuple):
        # decides which quadrant or the center the face is most in
        quadrants = ["Top Right", "Top Left", "Bottom Right", "Bottom Left"]
        if all(0 <= value <= 50 for value in pct_tuple):
            print("Center")
        elif all(value == 0 for value in pct_tuple):
            print("No Faces Detected")
        else:
            max_percentage_index = pct_tuple.index(max(pct_tuple))
            print(f"{quadrants[max_percentage_index]}")

    def draw_checklist(self):
        # Draw the checklist on the Pygame screen
        title_font = pygame.font.Font(None, 36)
        title_text = title_font.render("Selfie App Tester", True, (0, 0, 0))
        title_rect = title_text.get_rect(center=(self.WIDTH // 2, 30))
        self.screen.blit(title_text, title_rect)

        item_font = pygame.font.Font(None, 24)
        for i, item in enumerate(self.randomized_checklist):
            text = item_font.render(f"{item} {'(X)' if item in self.crossed_off_items else ''}", True, (0, 0, 0))
            text_rect = text.get_rect(center=(self.WIDTH // 2, 100 + i * 30))
            self.screen.blit(text, text_rect)

        # Draw subject's name input
        pygame.draw.rect(self.screen, (0, 0, 0), self.input_rect, 2)
        text_surface = self.input_font.render(self.input_text, True, (0, 0, 0))
        self.screen.blit(text_surface, (self.input_rect.x + 5, self.input_rect.y + 5))

    def cross_off_item(self):
        # Cross off an item in the checklist
        if self.randomized_checklist:
            item_to_cross_off = self.randomized_checklist.pop(0)
            self.crossed_off_items.add(item_to_cross_off)
            print(f"Crossed off: {item_to_cross_off}")

    def handle_input(self, event):
        # Handle keyboard input for subject's name
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                self.input_active = False
            elif event.key == pygame.K_BACKSPACE:
                self.input_text = self.input_text[:-1]
            else:
                self.input_text += event.unicode

    def run(self):
        # Runs selfie app loop
        while True:
            ret, frame = self.capture.read()
            self.screen.fill((255, 255, 255))  # Fill with a blank screen

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.capture.release()
                    pygame.quit()
                    cv2.destroyAllWindows()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        self.capture.release()
                        pygame.quit()
                        cv2.destroyAllWindows()
                        return
                    elif event.key == pygame.K_s:
                        self.take_photo()
                    elif event.key == pygame.K_i:
                        self.input_active = not self.input_active
                        self.input_text = ""

                # Handle text input
                if self.input_active:
                    self.handle_input(event)

            self.draw_checklist()
            self.print_quadrant(self.determine_quad_pct(frame))

            pygame.display.flip()

if __name__ == "__main__":
    app = SelfieApp()
    app.run()
