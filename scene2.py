# Selfie App V0.4
# Current Functionality: Take photos & locate the user's face.
# Authors: Tyler Cady, Conor Miller-Lynch
# Last Edit: 2024-01-18

import time

import cv2 as cv
import mediapipe as mp
import pygame
from ultralytics import YOLO
model = YOLO('yolov8n.pt')



CAMERA_ID = 0
FACE_DETECTION_MIN_CONFIDENCE = 0.2

IMAGE_FILE_EXTENSION = 'jpg'
SHUTTER_SOUND_FILE = 'resources/shutter.mp3'

GRIDLINES_COLOR_BGR = (255, 255, 255)
GRIDLINES_THICKNESS = 2
GRIDLINES_ALPHA = 0.5

FACE_BOX_LINE_COLOR_BGR = (0, 255, 0)
FACE_BOX_LINE_THICKNESS = 2

FACE_TOP_LEFT = 0
FACE_TOP_RIGHT = 1
FACE_BOTTOM_LEFT = 2
FACE_BOTTOM_RIGHT = 3
FACE_CENTER = 4
FACE_NONE = 5


class SceneApp:

    def __init__(self):
        """
        Initialize camera, face detection, and sound.
        """
        self.capture = cv.VideoCapture(CAMERA_ID)
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=FACE_DETECTION_MIN_CONFIDENCE)
        pygame.mixer.init()

    def take_photo(self):
        """
        Capture a frame, save the frame, and play a shutter sound.
        """
        ret, frame = self.capture.read()
        if ret:
            # Name photo with timestamp and add file extension.
            timestamp = time.strftime('%y%m%d%H%M%S')
            file_name = f'selfie_{timestamp}.{IMAGE_FILE_EXTENSION}'
            cv.imwrite(file_name, frame)
            print(f'Photo taken and saved as {file_name}')
            self.play_sound(SHUTTER_SOUND_FILE)

    def draw_grid(self, frame):
        """
        Draw a grid on frame that divides it into four quadrants and
        a center region.
        """
        h, w = frame.shape[:2]
        frame_copy = frame.copy()
        for line_start, line_end in (((w // 2, 0), (w // 2, h)),
                                     ((0, h // 2), (w, h // 2))):
            cv.line(frame_copy, line_start, line_end,
                    GRIDLINES_COLOR_BGR, GRIDLINES_THICKNESS)
        cv.rectangle(frame_copy, (w // 4, h // 4), (3 * w // 4, 3 * h // 4),
                     GRIDLINES_COLOR_BGR, GRIDLINES_THICKNESS)
        cv.addWeighted(frame_copy, GRIDLINES_ALPHA, frame,
                       1 - GRIDLINES_ALPHA, 0, frame)
    # Load YOLOv8n

    def detect_objects(self,frame):
        blob = cv.dnn.blobFromImage(frame, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        detections = net.forward()

        for detection in detections:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:  # Adjust confidence threshold as needed
                # Extract bounding box coordinates
                box = detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (x, y, w, h) = box.astype("int")

                # Draw bounding box and label
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f"{classes[class_id]}: {confidence:.2f}"
                cv.putText(frame, label, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return frame


    def draw_objects(self, frame):
        frame_with_objects = self.detect_objects(frame)
        return frame_with_objects


    def play_sound(self, file_path):
        """
        Play a sound from a file.
        """
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()

    def get_face_region(self, frame):
        """
        Determine which of the five regions the user's face is in.
        """
        h, w, _ = frame.shape
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)

        if results.detections:
            detection = results.detections[0]
            bboxC = detection.location_data.relative_bounding_box
            x, y, width, height = tuple(
                map(int, (bboxC.xmin * w,
                          bboxC.ymin * h,
                          bboxC.width * w,
                          bboxC.height * h)))
            total_face_area = width * height
            square1_area = ((w // 2) - x) * ((h // 2) - y)
            square2_area = (x + width - (w // 2)) * ((h // 2) - y)
            square3_area = ((w // 2) - x) * (y + height - (h // 2))
            square4_area = (x + width - (w // 2)) * (y + height - (h // 2))
            quadrant_pcts = tuple(((square_area / total_face_area) * 100)
                                  for square_area in (square1_area,
                                                      square2_area,
                                                      square3_area,
                                                      square4_area))

            # Determine and return the face location.
            if all(0 <= value <= 50 for value in quadrant_pcts):
                return FACE_CENTER
            elif all(value == 0 for value in quadrant_pcts):
                return FACE_NONE  # No faces detected
            else:
                quadrant_index = quadrant_pcts.index(max(quadrant_pcts))
                quadrants = [FACE_TOP_LEFT, FACE_TOP_RIGHT,
                             FACE_BOTTOM_LEFT, FACE_BOTTOM_RIGHT]
                return quadrants[quadrant_index]
        else:
            return FACE_NONE  # No faces detected

    def say(self, text): 
        print("this is a function stub")

    def listen(self):
        print("this is a function stub")

    def guide_user(self, loc, target):
        print("this is a function stub")

    def run(self):
        while True:
            ret, frame = self.capture.read()
            if ret:
                self.draw_grid(frame)
                frame_with_objects = self.draw_objects(frame)
                cv.imshow("Scene App", frame_with_objects)
            key = cv.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.take_photo()

        self.capture.release()
        cv.destroyAllWindows()



if __name__ == '__main__':
    app = SceneApp()
    app.run()
