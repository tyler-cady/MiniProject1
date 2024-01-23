# Selfie App V0.4
# Current Functionality: Take photos & locate the user's face.
# Authors: Tyler Cady, Conor Miller-Lynch
# Last Edit: 2024-01-18

import time

import cv2 as cv
import mediapipe as mp
import pygame
import speech_recognition as sr
import whisper
from gtts import gTTS

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

HELP_KEYWORDS = ['help', 'tutorial']
LEFT_KEYWORDS = ['left']
RIGHT_KEYWORDS = ['right']
TOP_KEYWORDS = ['top', 'upper']
BOTTOM_KEYWORDS = ['bottom', 'lower']
CENTER_KEYWORDS = ['center', 'middle']

WHISPER_MODEL = 'tiny.en'


class SelfieApp:

    last_position = 6
    time_of_last_hint = time.time()

    def __init__(self):
        """
        Initialize camera, face detection, microphone, and sound.
        """
        print('Initializing...')
        self.capture = cv.VideoCapture(CAMERA_ID)

        self.face_detection = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=FACE_DETECTION_MIN_CONFIDENCE)

        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()
        self.mic.__enter__()  # This is a hack to avoid using a with
        # statement
        self.recognizer.adjust_for_ambient_noise(self.mic, duration=1)
        whisper_model = getattr(self.recognizer, "whisper_model", {})
        whisper_model[WHISPER_MODEL] = whisper.load_model(WHISPER_MODEL)

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

    def draw_face_box(self, frame):
        """
        Draw a bounding box around the detected face.
        """
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw = frame.shape[:2]
                bbox = tuple(map(int, (bboxC.xmin * iw, bboxC.ymin * ih,
                                       bboxC.width * iw, bboxC.height * ih)))
                cv.rectangle(frame, bbox, FACE_BOX_LINE_COLOR_BGR,
                             FACE_BOX_LINE_THICKNESS)

    def play_sound(self, file_path):
        """
        Play a sound from a file.
        """
        self.channel.queue(pygame.mixer.Sound(file_path))

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

    def say(self, text, blocking=False):
        myobj = gTTS(text=text, lang='en', slow=False)
        myobj.save('resources/temp.mp3')
        print(text)
        self.play_sound('resources/temp.mp3')
        if blocking:
            while self.channel.get_busy():
                pygame.time.delay(100)

    def listen_for_command(self):
        print('Listening')
        audio = self.recognizer.listen(self.mic)
        print('Recognizing audio')
        spoken_text = self.recognizer.recognize_whisper(audio,
                                                        model=WHISPER_MODEL)
        print(f'You said "{spoken_text}"')
        spoken_text = spoken_text.lower()
        help_ = any(keyword in spoken_text for keyword in HELP_KEYWORDS)
        left = any(keyword in spoken_text for keyword in LEFT_KEYWORDS)
        right = any(keyword in spoken_text for keyword in RIGHT_KEYWORDS)
        top = any(keyword in spoken_text for keyword in TOP_KEYWORDS)
        bottom = any(keyword in spoken_text for keyword in BOTTOM_KEYWORDS)
        center = any(keyword in spoken_text for keyword in CENTER_KEYWORDS)
        if help_:
            self.tutorial()
            return None
        elif center and not any((left, right, top, bottom)):
            return FACE_CENTER
        elif left != right and top != bottom and not center:
            if left:
                if top:
                    return FACE_TOP_LEFT
                else:  # bottom
                    return FACE_BOTTOM_LEFT
            else:  # right
                if top:
                    return FACE_TOP_RIGHT
                else:  # bottom
                    return FACE_BOTTOM_RIGHT
        else:
            return None

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

    def tutorial(self):
        self.say('First, say the region where you want your face to be'
                 ' in the picture. Valid regions are "top left",'
                 ' "top right", "bottom left", and "bottom right".'
                 ' Then, follow the directions to move your face to the'
                 ' correct region. Press "S" to take a picture. Press'
                 ' "Q" to quit.', blocking=True)

    def main_menu(self):
        """
        Main menu of selfie app.
        """
        while True:
            self.say('Say a region or say "help" for help', blocking=True)
            target_region = self.listen_for_command()
            if target_region is not None:
                return target_region

    def run(self):
        """
        Run selfie app main loop.
        """
        while True:
            target_region = self.main_menu()
            if target_region == FACE_TOP_LEFT:
                self.say('Target region set to top left.')
            elif target_region == FACE_TOP_RIGHT:
                self.say('Target region set to top right.')
            elif target_region == FACE_BOTTOM_LEFT:
                self.say('Target region set to bottom left.')
            elif target_region == FACE_BOTTOM_RIGHT:
                self.say('Target region set to bottom right.')
            elif target_region == FACE_CENTER:
                self.say('Target region set to center.')
            while True:
                ret, frame = self.capture.read()
                if ret:
                    self.draw_grid(frame)
                    self.draw_face_box(frame)
                    cv.imshow("Selfie App", frame)
                    loc = self.get_face_region(frame)
                    self.guide_user(loc, target_region)
                key = cv.waitKey(1)
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.say('Taking photo...')
                    self.take_photo()
                    break

            key = cv.waitKey(1)
            if key == ord('q'):
                self.say('Quitting...')
                break

        self.mic.__exit__()  # This is a hack to avoid using a with
        # statement
        self.capture.release()
        cv.destroyAllWindows()


if __name__ == '__main__':
    app = SelfieApp()
    app.run()
