# Static Scene App V 1.1
# Current funcionality:
# Created by: Tyler Cady,
# Last Edit: 1.31.2024
import time

import cv2 as cv
import pygame
import speech_recognition as sr
from gtts import gTTS
from ultralytics import YOLO

# Load the model
model = YOLO('yolov8n.pt')

# Constants 
CAMERA_ID = 0
OBJ_DETECTION_THRESHOLD = 0.2
TIME_TO_COUNTDOWN = 2

IMAGE_FILE_EXTENSION = 'jpg'
SHUTTER_SOUND_FILE = 'resources/shutter.mp3'

GRIDLINES_COLOR_BGR = (255, 255, 255)
GRIDLINES_THICKNESS = 2
GRIDLINES_ALPHA = 0.5

FACE_BOX_LINE_COLOR_BGR = (0, 255, 0)
FACE_BOX_LINE_THICKNESS = 2

OBJ_TOP_LEFT = 0
OBJ_TOP_RIGHT = 1
OBJ_BOTTOM_LEFT = 2
OBJ_BOTTOM_RIGHT = 3
OBJ_CENTER = 4
OBJ_NONE = 5
QUIT = 6

HELP_KEYWORDS = ['help', 'tutorial']
QUIT_KEYWORDS = ['quit', 'exit']
LEFT_KEYWORDS = ['left']
RIGHT_KEYWORDS = ['right']
TOP_KEYWORDS = ['top', 'upper']
BOTTOM_KEYWORDS = ['bottom', 'lower']
CENTER_KEYWORDS = ['center', 'middle']

PHRASE_TIME_LIMIT = 2
WHISPER_MODEL = 'tiny.en'

class SceneApp:
    def __init__(self):
        # Init camera, face detection, and sound
        self.capture = cv2.VideoCapture(0)
        pygame.mixer.init()

        # Init object detection model
        self.model = YOLO('yolov8n.pt')
        
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()
        self.mic.__enter__()  # This is a hack to avoid using a with statement
        self.recognizer.adjust_for_ambient_noise(self.mic, duration=1)
        audio = self.recognizer.record(self.mic, 0.1)
        self.recognizer.recognize_whisper(audio, model=WHISPER_MODEL)  # Load Whisper model

        pygame.mixer.init()
        self.channel = pygame.mixer.Channel(0)

        myobj = gTTS(text='you are in the correct position',
                     lang='en', slow=False)
        myobj.save("resources/position.mp3")
        myobj = gTTS(text='you are out of view', lang='en', slow=False)
        myobj.save("resources/none.mp3")
        myobj = gTTS(text='move to the left', lang='en', slow=False)
        myobj.save("resources/left.mp3")
        myobj = gTTS(text='move to the right', lang='en', slow=False)
        myobj.save("resources/right.mp3")
        myobj = gTTS(text='move up', lang='en', slow=False)
        myobj.save("resources/up.mp3")
        myobj = gTTS(text='move down', lang='en', slow=False)
        myobj.save("resources/down.mp3")
        myobj = gTTS(text='move up and to the left', lang='en', slow=False)
        myobj.save("resources/up_left.mp3")
        myobj = gTTS(text='move down and to the left', lang='en', slow=False)
        myobj.save("resources/down_left.mp3")
        myobj = gTTS(text='move up and to the right', lang='en', slow=False)
        myobj.save("resources/up_right.mp3")
        myobj = gTTS(text='move down and to the right', lang='en', slow=False)
        myobj.save("resources/down_right.mp3")
    def take_photo(self):
        """
        Capture a frame, save the frame, and play a shutter sound.
        """
        ret, frame = self.capture.read()
        if ret:
            # Name photo with timestamp and add file extension.
            self.play_sound(SHUTTER_SOUND_FILE)
            timestamp = time.strftime('%Y-%m-%d_%H.%M.%S')
            file_name = f'selfie_{timestamp}.{IMAGE_FILE_EXTENSION}'
            cv.imwrite(file_name, frame)
            self.say(f'Photo taken and saved as {file_name}')
    def detect_objects(self, frame):
        results = model(frame)
        if isinstance(results, list) and len(results) > 0:
            result = results[0]
            if 'boxes' in result._keys:
                boxes = result.boxes
                for box in boxes:
                    class_id = int(box.cls)
                    class_name = result.names[class_id]
                    confidence = float(box.conf)
                    coords = [round(float(coord)) for coord in box.xyxy[0]]
                    if confidence > 0.5:
                        # Draw bounding box.
                        frame = cv2.rectangle(
                            frame, coords[0:2], coords[2:4], (0, 255, 0), 2)
                        # Add label.
                        frame = cv2.putText(frame, f"{class_name}: {confidence:.2f}",
                                            (coords[0], coords[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame
    # User Iteractions: 
    def listen_for_command(self):
        self.play_sound('resources/listening.mp3')
        print('Listening')
        audio = self.recognizer.listen(self.mic,
                                       phrase_time_limit=PHRASE_TIME_LIMIT)
        self.play_sound('resources/done_listening.mp3', queue=False)
        print('Recognizing audio')
        spoken_text = self.recognizer.recognize_whisper(audio,
                                                        model=WHISPER_MODEL)
        print(f'You said "{spoken_text}"')
        spoken_text = spoken_text.lower()
        help_ = any(keyword in spoken_text for keyword in HELP_KEYWORDS)
        quit_ = any(keyword in spoken_text for keyword in QUIT_KEYWORDS)
        left = any(keyword in spoken_text for keyword in LEFT_KEYWORDS)
        right = any(keyword in spoken_text for keyword in RIGHT_KEYWORDS)
        top = any(keyword in spoken_text for keyword in TOP_KEYWORDS)
        bottom = any(keyword in spoken_text for keyword in BOTTOM_KEYWORDS)
        center = any(keyword in spoken_text for keyword in CENTER_KEYWORDS)
        if help_:
            self.tutorial()
            return None
        elif quit_:
            return QUIT
        elif center and not any((left, right, top, bottom)):
            return OBJ_CENTER
        elif left != right and top != bottom and not center:
            if left:
                if top:
                    return OBJ_TOP_LEFT
                else:  # bottom
                    return OBJ_BOTTOM_LEFT
            else:  # right
                if top:
                    return OBJ_TOP_RIGHT
                else:  # bottom
                    return OBJ_BOTTOM_RIGHT
        else:
            return None
    def say(self, text, blocking=False):
        myobj = gTTS(text=text, lang='en', slow=False)
        myobj.save('resources/temp.mp3')
        print(text)
        self.play_sound('resources/temp.mp3')
        if blocking:
            while self.channel.get_busy():
                pygame.time.delay(100)
    def list_objects(self, frame):
        results = self.model(frame)
        if isinstance(results, list) and len(results) > 0:
            result = results[0]
            if 'boxes' in result._keys:
                boxes = result.boxes
                for box in boxes:
                    class_id = int(box.cls)
                    class_name = result.names[class_id]
                    confidence = float(box.conf)
                    coords = [round(float(coord)) for coord in box.xyxy[0]]
                    if confidence > 0.5:
                        detected = f"{class_name} detected"
                        self.say(detected)
    def tutorial(self):
        self.say("First, I will identify the objects in the frame."
                "Then, you will choose an object to photograph."
                "Next you will choose the desired position of the object in the frame."
                "Your options are: center, top left, top right, bottom left, bottom right."
                "Finally once you have the camera aimed correctly, I will take the photo and save it.")
    def play_sound(self, file_path):
        """
        Play a sound from a file.
        """
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
    def guide_user(self, loc, target):
        current_time = time.time()
        time_since_last_hint = current_time - self.time_of_last_hint
        if loc != self.last_position or time_since_last_hint >= 5:
            self.last_position = loc
            if loc == target:
                self.play_sound("resources/position.mp3")
            elif loc == FACE_TOP_LEFT:
                if (target == FACE_TOP_RIGHT):
                    self.play_sound("resources/left.mp3")
                if (target == FACE_CENTER):
                    self.play_sound("resources/down_left.mp3")
                if (target == FACE_BOTTOM_LEFT):
                    self.play_sound("resources/down.mp3")
                if (target == FACE_BOTTOM_RIGHT):
                    self.play_sound("resources/down_left.mp3")
            elif loc == FACE_TOP_RIGHT:
                if (target == FACE_TOP_LEFT):
                    self.play_sound("resources/right.mp3")
                if (target == FACE_CENTER):
                    self.play_sound("resources/down_right.mp3")
                if (target == FACE_BOTTOM_LEFT):
                    self.play_sound("resources/down_right.mp3")
                if (target == FACE_BOTTOM_RIGHT):
                    self.play_sound("resources/down.mp3")
            elif loc == FACE_CENTER:
                if (target == FACE_TOP_LEFT):
                    self.play_sound("resources/up_right.mp3")
                if (target == FACE_TOP_RIGHT):
                    self.play_sound("resources/up_left.mp3")
                if (target == FACE_BOTTOM_LEFT):
                    self.play_sound("resources/down_right.mp3")
                if (target == FACE_BOTTOM_RIGHT):
                    self.play_sound("resources/down_left.mp3")
            elif loc == FACE_BOTTOM_LEFT:
                if (target == FACE_TOP_LEFT):
                    self.play_sound("resources/up.mp3")
                if (target == FACE_TOP_RIGHT):
                    self.play_sound("resources/up_left.mp3")
                if (target == FACE_CENTER):
                    self.play_sound("resources/up_left.mp3")
                if (target == FACE_BOTTOM_RIGHT):
                    self.play_sound("resources/left.mp3")
            elif loc == FACE_BOTTOM_RIGHT:
                if (target == FACE_TOP_LEFT):
                    self.play_sound("resources/up_right.mp3")
                if (target == FACE_TOP_RIGHT):
                    self.play_sound("resources/up.mp3")
                if (target == FACE_CENTER):
                    self.play_sound("resources/up_right.mp3")
                if (target == FACE_BOTTOM_LEFT):
                    self.play_sound("resources/right.mp3")
            elif loc == FACE_NONE:
                self.play_sound("resources/none.mp3")
        self.time_of_last_hint = current_time
    def orange_error(self):
        #if camera detects orange, say "error orange detected, oranges don't exist"
        self.say("Error: Orange detected. Oranges don't exist.")
    def get_object_region(self, frame, object_coords):
        """
        Determine which region the detected object is in.
        """
        h, w, _ = frame.shape
        x, y, width, height = object_coords

        total_object_area = width * height
        square1_area = ((w // 2) - x) * ((h // 2) - y)
        square2_area = (x + width - (w // 2)) * ((h // 2) - y)
        square3_area = ((w // 2) - x) * (y + height - (h // 2))
        square4_area = (x + width - (w // 2)) * (y + height - (h // 2))

        quadrant_pcts = tuple(((square_area / total_object_area) * 100)
                            for square_area in (square1_area,
                                                square2_area,
                                                square3_area,
                                                square4_area))

        # Determine and return the object location.
        if all(0 <= value <= 50 for value in quadrant_pcts):
            return OBJ_CENTER
        elif all(value == 0 for value in quadrant_pcts):
            return OBJ_NONE  # No object detected
        else:
            quadrant_index = quadrant_pcts.index(max(quadrant_pcts))
            quadrants = [OBJ_TOP_LEFT, OBJ_TOP_RIGHT,
                        OBJ_BOTTOM_LEFT, OBJ_BOTTOM_RIGHT]
            return quadrants[quadrant_index]

    def mainMenu(self):
        while True:
            self.say("Welcome to SceneApp. Say a region or say help for a tutorial.")
            command = self.listen_for_command()
            if command is not None:
                return command
    def run(self):
        """
        Run selfie app main loop.
        """
        quit_ = False
        while True:
            target_region = self.main_menu()
            if target_region == OBJ_TOP_LEFT:
                self.say('Target region set to top left.', blocking=True)
            elif target_region == OBJ_TOP_RIGHT:
                self.say('Target region set to top right.', blocking=True)
            elif target_region == OBJ_BOTTOM_LEFT:
                self.say('Target region set to bottom left.', blocking=True)
            elif target_region == OBJ_BOTTOM_RIGHT:
                self.say('Target region set to bottom right.', blocking=True)
            elif target_region == OBJ_CENTER:
                self.say('Target region set to center.', blocking=True)
            elif target_region == QUIT:
                break
            time_in_target_region = -1
            while True:
                ret, frame = self.capture.read()
                if ret:
                    self.draw_grid(frame)
                    self.draw_face_box(frame)
                    cv.imshow("Scene App", frame)
                    loc = self.get_obj_region(frame)
                    self.guide_user(loc, target_region)
                if loc == target_region:
                    current_time = time.time()
                    if time_in_target_region == -1:
                        time_entered_target_region = current_time
                    time_in_target_region = (current_time
                                             - time_entered_target_region)
                else:  # not loc == target_region
                    time_in_target_region = -1
                key = cv.waitKey(1)
                if key == ord('q'):
                    quit_ = True
                    break
                elif key == ord('s') or time_in_target_region > TIME_TO_COUNTDOWN + 3:
                    end_time = time.time()
                    self.take_photo()
                elif time_in_target_region > TIME_TO_COUNTDOWN:
                    self.say('3... 2... 1...')
            if quit_:
                break

        self.say('Quitting')
        self.mic.__exit__(None, None, None)  # This is a hack to avoid
        # using a with statement
        self.capture.release()
        cv.destroyAllWindows()



if __name__ == "__main__":
    app = SceneApp()
    app.run()
