# Static Scene App V 0.1
# Current funcionality:
# Created by: Tyler Cady,
# Last Edit: 1.18.24
import cv2
import pygame
import time
from ultralytics import YOLO

# Load the model
model = YOLO('yolov8n.pt')


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

            # Detect objects and draw bounding boxes
            self.detect_objects(frame, file_name)

    def detect_objects(self, frame):
        results = model(frame)
        if isinstance(results, list) and len(results) > 0:
            result = results[0]
            if 'boxes' in result:
                boxes = result['boxes']
                for box in boxes:
                    class_id = int(box[4])
                    confidence = float(box[5])
                    if confidence > 0.5:
                        # Draw bounding box
                        frame = cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                        # Add label
                        frame = cv2.putText(frame, f"Class {class_id}: {confidence:.2f}", (int(box[0]), int(box[1])-10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return frame

    def detect_objects(self, frame):
    # Convert frame to RGB format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with the object detection model
    results = model(rgb_frame)

    if isinstance(results, list) and len(results) > 0:
        result = results[0]
        if 'boxes' in result:
            boxes = result['boxes']
            for box in boxes:
                class_id = int(box[4])
                confidence = float(box[5])
                if confidence > 0.5:
                    # Convert relative bounding box to absolute coordinates
                    ih, iw = frame.shape[:2]
                    xmin, ymin, width, height = box[:4]
                    bbox = tuple(map(int, (xmin * iw, ymin * ih, width * iw, height * ih)))

                    # Draw bounding box on the frame
                    frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
                    # Add label
                    frame = cv2.putText(frame, f"Class {class_id}: {confidence:.2f}", (bbox[0], bbox[1] - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame


    def play_sound(self, file_path):
        """
        Play a sound from a file.
        """
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()


    def run(self):
        try:
            while True:
                # Capture a frame from the camera
                ret, frame = self.capture.read()

                # Detect objects and draw bounding boxes live
                annotated_frame = self.detect_objects(frame)

                # Display the frame
                cv2.imshow("SceneApp", annotated_frame)

                # Check for keypress
                key = cv2.waitKey(1) & 0xFF

                # Take photo on 's' keypress
                if key == ord('s'):
                    self.take_photo()

                # Quit on 'q' keypress
                elif key == ord('q'):
                    break

        except Exception as e:
            print(f"An error occurred: {e}")

        finally:
            # Release the camera and close OpenCV window
            self.capture.release()
            cv2.destroyAllWindows()
            
if __name__ == "__main__":
    app = SceneApp()
    app.run()
