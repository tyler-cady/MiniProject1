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
START = 7

HELP_KEYWORDS = ['help', 'tutorial']
QUIT_KEYWORDS = ['quit', 'exit']
LEFT_KEYWORDS = ['left']
RIGHT_KEYWORDS = ['right']
TOP_KEYWORDS = ['top', 'upper']
BOTTOM_KEYWORDS = ['bottom', 'lower']
CENTER_KEYWORDS = ['center', 'middle']
START_KEYWORDS = ['start', 'begin']
ONE_KEYWORDS = ['one', '1']
TWO_KEYWORDS = ['two', '2']
THREE_KEYWORDS = ['three', '3']
FOUR_KEYWORDS = ['four', '4']
FIVE_KEYWORDS = ['five', '5']
SIX_KEYWORDS = ['six', '6']
SEVEN_KEYWORDS = ['seven', '7']
EIGHT_KEYWORDS = ['eight', '8']
NINE_KEYWORDS = ['nine', '9']

#OBJ_KEYWORDS = open("words.txt")

PHRASE_TIME_LIMIT = 2
WHISPER_MODEL = 'tiny.en'


class SceneApp:
    def __init__(self):

        self.last_position = 6
        self.time_of_last_hint = time.time()

        # Init camera, face detection, and sound
        self.capture = cv.VideoCapture(CAMERA_ID)
        pygame.mixer.init()

        # Init object detection model
        self.model = YOLO('yolov8n.pt')

        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()
        self.mic.__enter__()  # This is a hack to avoid using a with statement
        self.recognizer.adjust_for_ambient_noise(self.mic, duration=1)
        audio = self.recognizer.record(self.mic, 0.1)
        self.recognizer.recognize_whisper(
            audio, model=WHISPER_MODEL)  # Load Whisper model

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
                        frame = cv.rectangle(
                            frame, coords[0:2], coords[2:4], (0, 255, 0), 2)
                        # Add label.
                        frame = cv.putText(frame, f"{class_name}: {confidence:.2f}",
                                            (coords[0], coords[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame

    def find_target_obj(self, frame, target_object):
        results = model(frame)
        if isinstance(results, list) and len(results) > 0:
            result = results[0]
            if 'boxes' in result._keys:
                boxes = result.boxes
                for box in boxes:
                    class_id = int(box.cls)
                    class_name = result.names[class_id]
                    coords = [round(float(coord)) for coord in box.xyxy[0]]
                    if class_name == target_object:
                        return coords

        return -1

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
        start_ = any(keyword in spoken_text for keyword in START_KEYWORDS)
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
        elif start_:
            return START
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
    def listen_for_object(self):
        self.play_sound('resources/listening.mp3')
        print('Listening')
        audio = self.recognizer.listen(self.mic, phrase_time_limit=PHRASE_TIME_LIMIT)
        self.play_sound('resources/done_listening.mp3', queue=False)
        print('Recognizing audio')
        spoken_text = self.recognizer.recognize_whisper(audio, model=WHISPER_MODEL)
        print(f'You said "{spoken_text}"')

        spoken_text = spoken_text.lower()

        one_ = any(keyword in spoken_text for keyword in ONE_KEYWORDS)
        two_ = any(keyword in spoken_text for keyword in TWO_KEYWORDS)
        three_ = any(keyword in spoken_text for keyword in THREE_KEYWORDS)
        four_ = any(keyword in spoken_text for keyword in FOUR_KEYWORDS)
        five_ = any(keyword in spoken_text for keyword in FIVE_KEYWORDS)
        six_ = any(keyword in spoken_text for keyword in SIX_KEYWORDS)
        seven_ = any(keyword in spoken_text for keyword in SEVEN_KEYWORDS)
        eight_ = any(keyword in spoken_text for keyword in EIGHT_KEYWORDS)
        nine_ = any(keyword in spoken_text for keyword in NINE_KEYWORDS)

        if one_:
            return 1
        if two_:
            return 2
        if three_:
            return 3
        if four_:
            return 4
        if five_:
            return 5
        if six_:
            return 6
        if seven_:
            return 7
        if eight_:
            return 8
        if nine_:
            return 9

        #detected_objects = self.detect_objects(frame)  # Assuming frame is the current camera frame

        #if detected_objects:
        #    spoken_text = spoken_text.lower()
        #    for obj in detected_objects:
        #        if obj.lower() in spoken_text:
        #            return obj

            # If none of the detected objects match the spoken text
        #    self.say("Invalid object. Please try again.")
        #    return None

        return 0
    
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
                 "Finally once you have the camera aimed correctly, I will take the photo and save it.", blocking=True)

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

    def play_sound(self, file_path, queue=True):
        """
        Play a sound from a file.
        """
        sound = pygame.mixer.Sound(file_path)
        if queue:
            self.channel.queue(sound)
        else:
            self.channel.play(sound)

    def guide_user(self, loc, target):
        current_time = time.time()
        time_since_last_hint = current_time - self.time_of_last_hint
        if loc != self.last_position or time_since_last_hint >= 5:
            self.last_position = loc
            if loc == target:
                self.play_sound("resources/position.mp3")
            elif loc == OBJ_TOP_LEFT:
                if (target == OBJ_TOP_RIGHT):
                    self.play_sound("resources/left.mp3")
                if (target == OBJ_CENTER):
                    self.play_sound("resources/down_left.mp3")
                if (target == OBJ_BOTTOM_LEFT):
                    self.play_sound("resources/down.mp3")
                if (target == OBJ_BOTTOM_RIGHT):
                    self.play_sound("resources/down_left.mp3")
            elif loc == OBJ_TOP_RIGHT:
                if (target == OBJ_TOP_LEFT):
                    self.play_sound("resources/right.mp3")
                if (target == OBJ_CENTER):
                    self.play_sound("resources/down_right.mp3")
                if (target == OBJ_BOTTOM_LEFT):
                    self.play_sound("resources/down_right.mp3")
                if (target == OBJ_BOTTOM_RIGHT):
                    self.play_sound("resources/down.mp3")
            elif loc == OBJ_CENTER:
                if (target == OBJ_TOP_LEFT):
                    self.play_sound("resources/up_right.mp3")
                if (target == OBJ_TOP_RIGHT):
                    self.play_sound("resources/up_left.mp3")
                if (target == OBJ_BOTTOM_LEFT):
                    self.play_sound("resources/down_right.mp3")
                if (target == OBJ_BOTTOM_RIGHT):
                    self.play_sound("resources/down_left.mp3")
            elif loc == OBJ_BOTTOM_LEFT:
                if (target == OBJ_TOP_LEFT):
                    self.play_sound("resources/up.mp3")
                if (target == OBJ_TOP_RIGHT):
                    self.play_sound("resources/up_left.mp3")
                if (target == OBJ_CENTER):
                    self.play_sound("resources/up_left.mp3")
                if (target == OBJ_BOTTOM_RIGHT):
                    self.play_sound("resources/left.mp3")
            elif loc == OBJ_BOTTOM_RIGHT:
                if (target == OBJ_TOP_LEFT):
                    self.play_sound("resources/up_right.mp3")
                if (target == OBJ_TOP_RIGHT):
                    self.play_sound("resources/up.mp3")
                if (target == OBJ_CENTER):
                    self.play_sound("resources/up_right.mp3")
                if (target == OBJ_BOTTOM_LEFT):
                    self.play_sound("resources/right.mp3")
            elif loc == OBJ_NONE:
                self.play_sound("resources/none.mp3")
        self.time_of_last_hint = current_time

    def orange_error(self):
        # if camera detects orange, say "error orange detected, oranges don't exist"
        self.say("Error: Orange detected. Oranges don't exist.")

    def get_object_region(self, frame, object_coords):
        """
        Determine which region the detected object is in.
        """
        if object_coords == -1:
            return OBJ_NONE

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

    def choose_object(self, frame):
        """
        Choose an object to photograph.
        """
        self.list_objects(frame)

        detected_objects = []
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
                    if confidence > OBJ_DETECTION_THRESHOLD:
                        detected_objects.append(class_name)

        if not detected_objects:
            self.say("No objects detected. Please try again.", blocking=True)
            return None

        self.say("Detected objects in the scene:", blocking=True)
        for i, obj in enumerate(detected_objects, start=1):
            self.say(f"{i}. {obj}", blocking=True)

        while True:
            self.say(
                "Please choose a number for the object you'd like to photograph.", blocking=True)
            choice = self.listen_for_object()

            if 1 <= choice <= len(detected_objects):
                chosen_object = detected_objects[choice - 1]
                self.say(f"You've chosen to photograph {chosen_object}.", blocking=True)
                return chosen_object
            else:
                self.say("Invalid choice. Please choose a valid number.", blocking=True)

    def mainMenu(self):
        while True:
            self.say(
                "Welcome to SceneApp. Say start to start taking a picture, or say help for a tutorial.", blocking=True)
            command = self.listen_for_command()
            if command == START:
                return command

    def choose_region(self):
        while True:
            self.say(
                "Specify a region for the chosen object.", blocking=True)
            command = self.listen_for_command()
            if command is not None:
                if command != START:
                    return command

    def run(self):
        """
        Run scene app main loop.
        """

        quit_ = False
        while True:

            self.mainMenu()

            ret, frame = self.capture.read()
            target_object = self.choose_object(frame)

            target_region = self.choose_region()
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
                    cv.imshow("Scene App", frame)
                    coords = self.find_target_obj(frame, target_object)
                    print(coords)
                    loc = self.get_object_region(frame, coords)
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
