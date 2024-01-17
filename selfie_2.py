import cv2
import mediapipe as mp
import pygame


class SelfieApp:

    def __init__(self):
        self.capture = cv2.VideoCapture(0)
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.2)
        pygame.mixer.init()

    def take_photo(self):
        ret, frame = self.capture.read()
        if ret:
            cv2.imwrite("photo.jpg", frame)
            print("Photo taken and saved as photo.jpg")
            self.play_sound("resources/shutter.mp3")

    def draw_grid(self, frame):
        h, w, _ = frame.shape
        thickness = 2
        overlay = frame.copy()
        cv2.line(overlay, (w // 2, 0), (w // 2, h), (255, 255, 255), thickness)
        cv2.line(overlay, (0, h // 2), (w, h // 2), (255, 255, 255), thickness)
        cv2.rectangle(overlay, (w // 4, h // 4), (3 * w // 4, 3 * h // 4), (255, 255, 255), thickness)
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    def draw_face_box(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                bbox = tuple(map(int, (bboxC.xmin * iw, bboxC.ymin * ih, bboxC.width * iw, bboxC.height * ih)))
                cv2.rectangle(frame, bbox, (0, 255, 0), 2)

    def play_sound(self, file_path):
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()

    def determine_quad_pct(self, frame):
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
            pct_tuple = tuple(((square_area / total_face_area) * 100) for square_area in (square1_area, square2_area, square3_area, square4_area))
            return pct_tuple
        else:
            return (0, 0, 0, 0)

    def print_quadrant(self, pct_tuple):
        quadrants = ["Top Left", "Top Right", "Bottom Left", "Bottom Right"]
        if all(0 <= value <= 50 for value in pct_tuple):
            print("The face is most in Center")
        elif all(value == 0 for value in pct_tuple):
            print("No Faces Detected")
        else:
            max_percentage_index = pct_tuple.index(max(pct_tuple))
            print(f"The face is most in {quadrants[max_percentage_index]}")

    def run(self):
        while True:
            ret, frame = self.capture.read()
            self.draw_grid(frame)
            self.draw_face_box(frame)
            cv2.imshow("Selfie App", frame)
            self.print_quadrant(self.determine_quad_pct(frame))
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.take_photo()

        self.capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = SelfieApp()
    app.run()
