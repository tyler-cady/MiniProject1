# Selfie App V0.4
# Current Functionality: Take photos & locate the user's face.
# Authors: Tyler Cady, Conor Miller-Lynch
# Last Edit: 2024-01-18

import time

import cv2 as cv
import mediapipe as mp
import pygame
import speech_recognition as sr
from gtts import gTTS

TEST_MODE = False

CAMERA_ID = 0
FACE_DETECTION_MIN_CONFIDENCE = 0.2
TIME_BETWEEN_HINTS = 2
TIME_TO_COUNTDOWN = 2
COUNTDOWN_SECONDS = 2.5

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
        myobj = gTTS(text='move left', lang='en', slow=False)
        myobj.save("resources/left.mp3")
        myobj = gTTS(text='move right', lang='en', slow=False)
        myobj.save("resources/right.mp3")
        myobj = gTTS(text='move up', lang='en', slow=False)
        myobj.save("resources/up.mp3")
        myobj = gTTS(text='move down', lang='en', slow=False)
        myobj.save("resources/down.mp3")
        myobj = gTTS(text='move up and left', lang='en', slow=False)
        myobj.save("resources/up_left.mp3")
        myobj = gTTS(text='move down and left', lang='en', slow=False)
        myobj.save("resources/down_left.mp3")
        myobj = gTTS(text='move up and right', lang='en', slow=False)
        myobj.save("resources/up_right.mp3")
        myobj = gTTS(text='move down and right', lang='en', slow=False)
        myobj.save("resources/down_right.mp3")

    def take_photo(self):
        """
        Capture a frame, save the frame, and play a shutter sound.
        """
        ret, frame = self.capture.read()
        if ret:
            # Name photo with timestamp and add file extension.
            self.play_sound(SHUTTER_SOUND_FILE, queue=False)
            timestamp = time.strftime('%Y-%m-%d_%H.%M.%S')
            file_name = f'selfie_{timestamp}.{IMAGE_FILE_EXTENSION}'
            cv.imwrite(file_name, frame)
            self.say(f'Photo taken and saved')

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

    def play_sound(self, file_path, queue=True):
        """
        Play a sound from a file.
        """
        sound = pygame.mixer.Sound(file_path)
        if queue:
            self.channel.queue(sound)
        else:
            self.channel.play(sound)

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

    def say(self, text, blocking=False, queue=True):
        myobj = gTTS(text=text, lang='en', slow=False)
        myobj.save('resources/temp.mp3')
        print(text)
        self.play_sound('resources/temp.mp3', queue=queue)
        if blocking:
            while self.channel.get_busy():
                pygame.time.delay(100)

    def listen_for_command(self):
        self.play_sound('resources/listening_new.mp3')
        print('Listening')
        audio = self.recognizer.listen(self.mic,
                                       phrase_time_limit=PHRASE_TIME_LIMIT)
        self.play_sound('resources/done_listening_new.mp3', queue=False)
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
        if ((loc != self.last_position
            or (time_since_last_hint >= TIME_BETWEEN_HINTS and loc != target))
            and not self.countdown_begun):
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
                 ' in the picture. Valid regions are "top left", "top'
                 ' right", "center", "bottom left", and "bottom right".'
                 ' Then, follow the directions to move your face to the'
                 ' specified region. A countdown will begin'
                 ' automatically after you have been in the specified'
                 f' region for {TIME_TO_COUNTDOWN} seconds. When the'
                 ' countdown is finished, a picture will be taken.'
                 ' After a picture has been taken, you may specify a'
                 ' different region and take another picture, say'
                 ' "quit" to quit the application, or say "help" to'
                 ' access this information. You may also press the "S"'
                 ' key to take a picture at any time or the "Q" key to'
                 ' quit at any time.', blocking=True)

    def main_menu(self):
        """
        Main menu of selfie app.
        """
        if TEST_MODE:
            self.participant_id = input('Enter Particpant ID: ')
            self.start_time = time.time()
        while True:
            self.say('Say a region or say "help" for help', blocking=True)
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
            if target_region == FACE_TOP_LEFT:
                self.say('Target region set to top left', blocking=True)
            elif target_region == FACE_TOP_RIGHT:
                self.say('Target region set to top right', blocking=True)
            elif target_region == FACE_BOTTOM_LEFT:
                self.say('Target region set to bottom left', blocking=True)
            elif target_region == FACE_BOTTOM_RIGHT:
                self.say('Target region set to bottom right', blocking=True)
            elif target_region == FACE_CENTER:
                self.say('Target region set to center', blocking=True)
            elif target_region == QUIT:
                break
            time_in_target_region = -1
            self.countdown_begun = False
            while True:
                ret, frame = self.capture.read()
                if ret:
                    self.draw_grid(frame)
                    self.draw_face_box(frame)
                    cv.imshow("Selfie App", frame)
                    loc = self.get_face_region(frame)
                    self.guide_user(loc, target_region)
                if loc == target_region:
                    current_time = time.time()
                    if time_in_target_region == -1:
                        time_entered_target_region = current_time
                    time_in_target_region = (current_time
                                             - time_entered_target_region)
                else:  # not loc == target_region
                    time_in_target_region = -1
                    self.countdown_begun = False
                key = cv.waitKey(1)
                if key == ord('q'):
                    quit_ = True
                    break
                elif key == ord('s') or time_in_target_region > TIME_TO_COUNTDOWN + COUNTDOWN_SECONDS:
                    end_time = time.time()
                    self.take_photo()
                    if TEST_MODE:
                        with open(f'data_{self.participant_id}.csv', 'a') as f:
                            f.write(
                                f'{self.participant_id},{target_region},{end_time - self.start_time}\n')
                    break
                elif time_in_target_region > TIME_TO_COUNTDOWN:
                    if not self.countdown_begun:
                        self.say('3... 2... 1...', queue=False)
                        self.countdown_begun = True
            if quit_:
                break

        self.say('Quitting', blocking=True)
        self.mic.__exit__(None, None, None)  # This is a hack to avoid
        # using a with statement
        self.capture.release()
        cv.destroyAllWindows()


if __name__ == '__main__':
    app = SelfieApp()
    app.run()
