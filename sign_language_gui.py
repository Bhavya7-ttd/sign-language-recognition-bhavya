# Importing Libraries
import numpy as np
import math
import cv2
import os, sys
import traceback
import pyttsx3
from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
from string import ascii_uppercase
import tkinter as tk
from PIL import Image, ImageTk
import threading

try:
    import enchant
    ddd = enchant.Dict("en-US")
except Exception as e:
    print("WARNING: pyenchant dictionary not available:", e)
    ddd = None

offset = 29

os.environ["THEANO_FLAGS"] = "device=cuda, assert_no_cpu_op=True"


# Application :

class Application:

    def __init__(self):
        self.vs = cv2.VideoCapture(0)
        self.current_image = None
        self.model = load_model('cnn8grps_rad1_model.h5')
        self.staged_char = None
        
        # Safe text-to-speech initialization using non-blocking manual loop
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", 120)
            voices = self.engine.getProperty("voices")
            if voices:
                self.engine.setProperty("voice", voices[0].id)
            self.engine.startLoop(False)
            self.tts_available = True
        except Exception as e:
            print("WARNING: Text-to-speech engine failed to initialize:", e)
            self.tts_available = False

        self.ct = {}
        self.ct['blank'] = 0
        self.blank_flag = 0
        self.space_flag=False
        self.next_flag=True
        self.prev_char=""
        self.count=-1
        self.ten_prev_char=[]
        for i in range(10):
            self.ten_prev_char.append(" ")

        for i in ascii_uppercase:
            self.ct[i] = 0

        # Hand detector setup (configured to match test_hand.py)
        self.hd = HandDetector(
            staticMode=False,
            maxHands=1,
            modelComplexity=1,
            detectionCon=0.5,
            minTrackCon=0.5
        )

        # Debouncing and Stable Prediction variables
        self.stability_threshold = 15
        self.prev_raw_char = "blank"
        self.consecutive_frames = 0
        self.last_appended_char = "blank"

        # Local words fallback
        self.local_words = ["HELLO", "HELP", "HELMET", "HELICOPTER", "WORLD", "WORK", "WORD", "WELCOME", "SIGN", "LANGUAGE", "PROJECT", "TEST", "PLEASE", "THANK", "YOU", "DEAF", "GOOD", "MORNING", "AFTERNOON", "NIGHT", "FRIEND", "FAMILY", "HOME", "SCHOOL", "YES", "NO", "NAME", "WHAT", "HOW", "WHO", "WHY", "WHERE", "WHEN"]

        print("Loaded model from disk")

        # Root Window Setup
        self.root = tk.Tk()
        self.root.title("Sign Language To Text Conversion")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.root.geometry("1200x820")
        self.root.configure(bg="#1e1e24")

        # Premium Dark Styles
        bg_main = "#1e1e24"
        bg_card = "#2a2a35"
        fg_text = "#ffffff"
        fg_accent = "#3a86ff"
        fg_success = "#06d6a0"
        fg_danger = "#ef476f"
        
        font_title = ("Helvetica", 24, "bold")
        font_header = ("Helvetica", 14, "bold")
        font_body = ("Helvetica", 12)
        font_display = ("Consolas", 20, "bold")

        # 1. Header Frame
        header_frame = tk.Frame(self.root, bg=bg_main, pady=10)
        header_frame.pack(fill=tk.X)
        
        title_label = tk.Label(header_frame, text="Sign Language To Text Conversion", font=font_title, fg=fg_text, bg=bg_main)
        title_label.pack()

        # 3. Control Panel (Sentence, Suggestions, Buttons)
        control_frame = tk.Frame(self.root, bg=bg_card, bd=1, relief=tk.SOLID, padx=20, pady=15)
        control_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=20, pady=20)

        # 2. Main Workspace (Video + Skeleton Canvas)
        workspace_frame = tk.Frame(self.root, bg=bg_main, padx=20, pady=10)
        workspace_frame.pack(fill=tk.BOTH, expand=True)

        # Video Panel Frame
        video_frame = tk.LabelFrame(workspace_frame, text=" Camera Feed (Hand & Landmarks) ", font=font_header, fg=fg_text, bg=bg_card, bd=2, relief=tk.GROOVE)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.panel = tk.Label(video_frame, bg="#000000")
        self.panel.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Skeleton Canvas Frame
        skeleton_frame = tk.LabelFrame(workspace_frame, text=" Model Input Canvas (400x400) ", font=font_header, fg=fg_text, bg=bg_card, bd=2, relief=tk.GROOVE)
        skeleton_frame.pack(side=tk.RIGHT, fill=tk.NONE, expand=False, padx=10, pady=5)
        
        self.panel2 = tk.Label(skeleton_frame, bg="#ffffff", width=400, height=400)
        self.panel2.pack(padx=5, pady=5)

        # Detected Sign and Configuration row
        info_row = tk.Frame(control_frame, bg=bg_card)
        info_row.pack(fill=tk.X, pady=5)

        char_label = tk.Label(info_row, text="Detected Sign: ", font=font_header, fg=fg_accent, bg=bg_card)
        char_label.pack(side=tk.LEFT)

        self.panel3 = tk.Label(info_row, text="-", font=("Helvetica", 20, "bold"), fg=fg_success, bg=bg_card)
        self.panel3.pack(side=tk.LEFT, padx=10)

        # Stability Slider
        slider_frame = tk.Frame(info_row, bg=bg_card)
        slider_frame.pack(side=tk.RIGHT)
        
        slider_label = tk.Label(slider_frame, text="Stability (Frames): ", font=font_body, fg=fg_text, bg=bg_card)
        slider_label.pack(side=tk.LEFT)
        
        self.stability_slider = tk.Scale(slider_frame, from_=5, to=30, orient=tk.HORIZONTAL, bg=bg_card, fg=fg_text,
                                         highlightthickness=0, troughcolor="#444455", activebackground=fg_accent)
        self.stability_slider.set(self.stability_threshold)
        self.stability_slider.pack(side=tk.LEFT, padx=5)

        # Sentence Box Row
        sentence_row = tk.Frame(control_frame, bg=bg_card, pady=5)
        sentence_row.pack(fill=tk.X)

        sentence_label = tk.Label(sentence_row, text="Current Text: ", font=font_header, fg=fg_text, bg=bg_card)
        sentence_label.pack(side=tk.LEFT)

        self.panel5 = tk.Label(sentence_row, text="", font=font_display, fg=fg_success, bg="#18181f", anchor=tk.W, justify=tk.LEFT, padx=10, pady=5, relief=tk.SUNKEN, bd=1)
        self.panel5.pack(fill=tk.X, expand=True, padx=10)

        # Suggestions Row
        suggestions_row = tk.Frame(control_frame, bg=bg_card, pady=10)
        suggestions_row.pack(fill=tk.X)

        sug_label = tk.Label(suggestions_row, text="Suggestions: ", font=font_body, fg=fg_text, bg=bg_card)
        sug_label.pack(side=tk.LEFT)

        # Suggestion buttons
        self.b1 = tk.Button(suggestions_row, text=" ", font=font_body, bg="#3e3e4f", fg=fg_text, activebackground=fg_accent, activeforeground=fg_text, bd=0, padx=15, pady=5, command=self.action1)
        self.b1.pack(side=tk.LEFT, padx=5)
        
        self.b2 = tk.Button(suggestions_row, text=" ", font=font_body, bg="#3e3e4f", fg=fg_text, activebackground=fg_accent, activeforeground=fg_text, bd=0, padx=15, pady=5, command=self.action2)
        self.b2.pack(side=tk.LEFT, padx=5)

        self.b3 = tk.Button(suggestions_row, text=" ", font=font_body, bg="#3e3e4f", fg=fg_text, activebackground=fg_accent, activeforeground=fg_text, bd=0, padx=15, pady=5, command=self.action3)
        self.b3.pack(side=tk.LEFT, padx=5)

        self.b4 = tk.Button(suggestions_row, text=" ", font=font_body, bg="#3e3e4f", fg=fg_text, activebackground=fg_accent, activeforeground=fg_text, bd=0, padx=15, pady=5, command=self.action4)
        self.b4.pack(side=tk.LEFT, padx=5)

        # Action Buttons Row
        actions_row = tk.Frame(control_frame, bg=bg_card, pady=5)
        actions_row.pack(fill=tk.X)

        self.speak = tk.Button(actions_row, text="Speak 🔊", font=font_body, bg=fg_accent, fg=fg_text, activebackground="#1e60ff", activeforeground=fg_text, bd=0, padx=20, pady=8, command=self.speak_fun)
        self.speak.pack(side=tk.RIGHT, padx=5)

        self.clear = tk.Button(actions_row, text="Clear ❌", font=font_body, bg=fg_danger, fg=fg_text, activebackground="#df375f", activeforeground=fg_text, bd=0, padx=20, pady=8, command=self.clear_fun)
        self.clear.pack(side=tk.RIGHT, padx=5)

        self.btn_space = tk.Button(actions_row, text="Space ␣", font=font_body, bg="#3e3e4f", fg=fg_text, activebackground=fg_accent, activeforeground=fg_text, bd=0, padx=20, pady=8, command=self.space_fun)
        self.btn_space.pack(side=tk.RIGHT, padx=5)

        self.btn_backspace = tk.Button(actions_row, text="Backspace ⌫", font=font_body, bg="#3e3e4f", fg=fg_text, activebackground=fg_accent, activeforeground=fg_text, bd=0, padx=20, pady=8, command=self.backspace_fun)
        self.btn_backspace.pack(side=tk.RIGHT, padx=5)

        self.str = " "
        self.ccc = 0
        self.word = " "
        self.current_symbol = "-"
        self.photo = "Empty"

        self.word1 = " "
        self.word2 = " "
        self.word3 = " "
        self.word4 = " "

        self.video_loop()

    def video_loop(self):
        try:
            ok, frame = self.vs.read()
            if not ok or frame is None:
                print("Could not read frame from webcam.")
                self.root.after(10, self.video_loop)
                return

            # Mirror frame
            cv2image = cv2.flip(frame, 1)
            cv2image_copy = np.array(cv2image)

            # Detect hand on mirrored frame using draw=True and flipType=False (matches test_hand.py)
            hands, cv2image = self.hd.findHands(cv2image, draw=True, flipType=False)
            
            # Convert frame to RGB for Tkinter display
            cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGB)
            self.current_image = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=self.current_image)
            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)

            raw_char = "blank"

            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                
                # Safe crop bounds checking
                frame_h, frame_w = cv2image_copy.shape[:2]
                y1 = max(0, y - offset)
                y2 = min(frame_h, y + h + offset)
                x1 = max(0, x - offset)
                x2 = min(frame_w, x + w + offset)

                if (y2 - y1) > 0 and (x2 - x1) > 0:
                    pts_original = hand['lmList']
                    
                    # Mathematically translate landmarks to cropped coordinate system
                    self.pts = []
                    for pt in pts_original:
                        px = pt[0] - x1
                        py = pt[1] - y1
                        pz = pt[2]
                        self.pts.append([px, py, pz])

                    self.pts = [[int(pt[0]), int(pt[1]), int(pt[2])] for pt in self.pts]

                    # Centering calculation offsets for 400x400 canvas (matching training data collection)
                    os_val = int(((400 - w) // 2) - 15)
                    os1_val = int(((400 - h) // 2) - 15)

                    # Hand orientation handling (Left hand mirroring)
                    # Mirror left-hand coordinates horizontally so they behave like a right hand
                    is_left_hand = (hand.get("type", "Right") == "Left")
                    if is_left_hand:
                        for i in range(21):
                            canvas_x = self.pts[i][0] + os_val
                            mirrored_canvas_x = 400 - canvas_x
                            self.pts[i][0] = mirrored_canvas_x - os_val

                    # Generate dynamic white background
                    white = np.ones((400, 400, 3), np.uint8) * 255

                    # Draw hand skeleton lines
                    for t in range(0, 4, 1):
                        cv2.line(white, (self.pts[t][0] + os_val, self.pts[t][1] + os1_val), (self.pts[t + 1][0] + os_val, self.pts[t + 1][1] + os1_val),
                                 (0, 255, 0), 3)
                    for t in range(5, 8, 1):
                        cv2.line(white, (self.pts[t][0] + os_val, self.pts[t][1] + os1_val), (self.pts[t + 1][0] + os_val, self.pts[t + 1][1] + os1_val),
                                 (0, 255, 0), 3)
                    for t in range(9, 12, 1):
                        cv2.line(white, (self.pts[t][0] + os_val, self.pts[t][1] + os1_val), (self.pts[t + 1][0] + os_val, self.pts[t + 1][1] + os1_val),
                                 (0, 255, 0), 3)
                    for t in range(13, 16, 1):
                        cv2.line(white, (self.pts[t][0] + os_val, self.pts[t][1] + os1_val), (self.pts[t + 1][0] + os_val, self.pts[t + 1][1] + os1_val),
                                 (0, 255, 0), 3)
                    for t in range(17, 20, 1):
                        cv2.line(white, (self.pts[t][0] + os_val, self.pts[t][1] + os1_val), (self.pts[t + 1][0] + os_val, self.pts[t + 1][1] + os1_val),
                                 (0, 255, 0), 3)
                    
                    cv2.line(white, (self.pts[5][0] + os_val, self.pts[5][1] + os1_val), (self.pts[9][0] + os_val, self.pts[9][1] + os1_val), (0, 255, 0), 3)
                    cv2.line(white, (self.pts[9][0] + os_val, self.pts[9][1] + os1_val), (self.pts[13][0] + os_val, self.pts[13][1] + os1_val), (0, 255, 0), 3)
                    cv2.line(white, (self.pts[13][0] + os_val, self.pts[13][1] + os1_val), (self.pts[17][0] + os_val, self.pts[17][1] + os1_val), (0, 255, 0), 3)
                    cv2.line(white, (self.pts[0][0] + os_val, self.pts[0][1] + os1_val), (self.pts[5][0] + os_val, self.pts[5][1] + os1_val), (0, 255, 0), 3)
                    cv2.line(white, (self.pts[0][0] + os_val, self.pts[0][1] + os1_val), (self.pts[17][0] + os_val, self.pts[17][1] + os1_val), (0, 255, 0), 3)

                    # Draw red circles at joints
                    for i in range(21):
                        cv2.circle(white, (self.pts[i][0] + os_val, self.pts[i][1] + os1_val), 2, (0, 0, 255), 1)

                    # Predict sign
                    self.predict(white)
                    raw_char = self.current_symbol

                    # Display skeleton canvas image in panel2
                    self.current_image2 = Image.fromarray(white)
                    imgtk2 = ImageTk.PhotoImage(image=self.current_image2)
                    self.panel2.imgtk = imgtk2
                    self.panel2.config(image=imgtk2)

            # If no hand was found or crop was invalid, reset symbols
            if not hands:
                self.current_symbol = "-"
                # Draw empty white canvas
                empty_white = np.ones((400, 400, 3), np.uint8) * 255
                self.current_image2 = Image.fromarray(empty_white)
                imgtk2 = ImageTk.PhotoImage(image=self.current_image2)
                self.panel2.imgtk = imgtk2
                self.panel2.config(image=imgtk2)

            self.panel3.config(text=self.current_symbol)

            # Debouncing state machine for stable character prediction
            if hasattr(self, 'stability_slider'):
                self.stability_threshold = self.stability_slider.get()

            # Responsive threshold for control gestures like "next"
            current_threshold = self.stability_threshold
            if raw_char in ["next", "Next"]:
                current_threshold = 3  # Highly responsive for Next gesture!

            if raw_char == self.prev_raw_char:
                self.consecutive_frames += 1
            else:
                self.consecutive_frames = 1
                self.prev_raw_char = raw_char

            if self.consecutive_frames >= current_threshold:
                stable_char = raw_char
                if stable_char != self.last_appended_char:
                    if stable_char == "blank":
                        self.last_appended_char = "blank"
                    elif stable_char == " ":
                        self.str += " "
                        self.last_appended_char = stable_char
                        self.update_suggestions()
                    elif stable_char in ["Backspace", "backspace"]:
                        if len(self.str) > 0:
                            self.str = self.str[:-1]
                        self.last_appended_char = stable_char
                        self.update_suggestions()
                    elif stable_char in ["next", "Next"]:
                        print("PREDICTION: Next")
                        print("NEXT CONFIRMED")
                        if self.staged_char is not None:
                            print(f"COMMITTING: {self.staged_char}")
                            self.str += str(self.staged_char)
                            print(f"TEXT NOW: {self.str}")
                            self.speak_text(self.staged_char)
                            self.staged_char = None
                        self.last_appended_char = stable_char
                        self.update_suggestions()
                    else:
                        if self.staged_char != stable_char:
                            self.staged_char = stable_char
                            print(f"PREDICTION: {self.staged_char}")
                        self.last_appended_char = stable_char
                        self.update_suggestions()

            # Pump TTS engine events on the main thread loop
            if self.tts_available:
                try:
                    self.engine.iterate()
                except Exception as e:
                    print("TTS Iterate Error:", e)

            self.panel5.config(text=self.str)
            self.root.after(10, self.video_loop)
        except Exception as e:
            print("Loop Error:", e)
            self.root.after(10, self.video_loop)

    def distance(self, x, y):
        return math.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))

    def replace_last_word(self, replacement):
        # Splitting keeping empty strings so we preserve spaces
        words = self.str.split(" ")
        if words:
            words[-1] = replacement.upper()
            self.str = " ".join(words) + " "
            self.staged_char = None
            self.update_suggestions()
            self.speak_text(replacement)

    def action1(self):
        if self.word1.strip():
            self.replace_last_word(self.word1)

    def action2(self):
        if self.word2.strip():
            self.replace_last_word(self.word2)

    def action3(self):
        if self.word3.strip():
            self.replace_last_word(self.word3)

    def action4(self):
        if self.word4.strip():
            self.replace_last_word(self.word4)

    def speak_text(self, text):
        if self.tts_available and text.strip():
            try:
                print(f"VOICE REQUEST: {text}")
                print(f"VOICE STARTED: {text}")
                self.engine.say(text)
                print(f"VOICE FINISHED: {text}")
            except Exception as e:
                print("TTS speak error:", e)

    def speak_fun(self):
        if self.str.strip():
            self.speak_text(self.str)

    def clear_fun(self):
        self.str = ""
        self.word1 = " "
        self.word2 = " "
        self.word3 = " "
        self.word4 = " "
        self.update_suggestions()

    def space_fun(self):
        self.str += " "
        self.update_suggestions()

    def backspace_fun(self):
        if len(self.str) > 0:
            self.str = self.str[:-1]
        self.update_suggestions()

    def get_local_suggestions(self, prefix):
        prefix = prefix.upper()
        candidates = [w for w in self.local_words if w.startswith(prefix)]
        candidates.sort(key=len)
        return candidates[:4]

    def update_suggestions(self):
        text_to_check = self.str
        if self.staged_char:
            text_to_check += self.staged_char
        words = text_to_check.strip().split(" ")
        if words and len(words[-1]) > 0:
            word = words[-1]
            self.word = word
            suggestions = []
            if ddd:
                try:
                    if not ddd.check(word):
                        suggestions = ddd.suggest(word)
                    else:
                        suggestions = [word] + ddd.suggest(word)
                except Exception:
                    pass
            
            # Fallback to local dictionary if suggestions are empty
            if not suggestions:
                suggestions = self.get_local_suggestions(word)

            self.word1 = suggestions[0] if len(suggestions) >= 1 else " "
            self.word2 = suggestions[1] if len(suggestions) >= 2 else " "
            self.word3 = suggestions[2] if len(suggestions) >= 3 else " "
            self.word4 = suggestions[3] if len(suggestions) >= 4 else " "
        else:
            self.word = " "
            self.word1 = " "
            self.word2 = " "
            self.word3 = " "
            self.word4 = " "

        self.b1.config(text=self.word1)
        self.b2.config(text=self.word2)
        self.b3.config(text=self.word3)
        self.b4.config(text=self.word4)

    def predict(self, test_image):
        white = test_image
        white = white.reshape(1, 400, 400, 3)
        prob = np.array(self.model.predict(white, verbose=0)[0], dtype='float32')
        ch1 = np.argmax(prob, axis=0)
        prob[ch1] = 0
        ch2 = np.argmax(prob, axis=0)
        prob[ch2] = 0
        ch3 = np.argmax(prob, axis=0)
        prob[ch3] = 0

        pl = [ch1, ch2]

        # condition for [Aemnst]
        l = [[5, 2], [5, 3], [3, 5], [3, 6], [3, 0], [3, 2], [6, 4], [6, 1], [6, 2], [6, 6], [6, 7], [6, 0], [6, 5],
             [4, 1], [1, 0], [1, 1], [6, 3], [1, 6], [5, 6], [5, 1], [4, 5], [1, 4], [1, 5], [2, 0], [2, 6], [4, 6],
             [1, 0], [5, 7], [1, 6], [6, 1], [7, 6], [2, 5], [7, 1], [5, 4], [7, 0], [7, 5], [7, 2]]
        if pl in l:
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][
                1]):
                ch1 = 0
                # print("00000")

        # condition for [o][s]
        l = [[2, 2], [2, 1]]
        if pl in l:
            if (self.pts[5][0] < self.pts[4][0]):
                ch1 = 0
                print("++++++++++++++++++")
                # print("00000")

        # condition for [c0][aemnst]
        l = [[0, 0], [0, 6], [0, 2], [0, 5], [0, 1], [0, 7], [5, 2], [7, 6], [7, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[4][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][
                0] and self.pts[0][0] > self.pts[20][0]) and self.pts[5][0] > self.pts[4][0]:
                ch1 = 2
                # print("22222")

        # condition for [c0][aemnst]
        l = [[6, 0], [6, 6], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[8], self.pts[16]) < 52:
                ch1 = 2
                # print("22222")


        # condition for [gh][bdfikruvw]
        l = [[1, 4], [1, 5], [1, 6], [1, 3], [1, 0]]
        pl = [ch1, ch2]

        if pl in l:
            if self.pts[6][1] > self.pts[8][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1] and self.pts[0][0] < self.pts[8][
                0] and self.pts[0][0] < self.pts[12][0] and self.pts[0][0] < self.pts[16][0] and self.pts[0][0] < self.pts[20][0]:
                ch1 = 3
                print("33333c")



        # con for [gh][l]
        l = [[4, 6], [4, 1], [4, 5], [4, 3], [4, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[4][0] > self.pts[0][0]:
                ch1 = 3
                print("33333b")

        # con for [gh][pqz]
        l = [[5, 3], [5, 0], [5, 7], [5, 4], [5, 2], [5, 1], [5, 5]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[2][1] + 15 < self.pts[16][1]:
                ch1 = 3
                print("33333a")

        # con for [l][x]
        l = [[6, 4], [6, 1], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[4], self.pts[11]) > 55:
                ch1 = 4
                # print("44444")

        # con for [l][d]
        l = [[1, 4], [1, 6], [1, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.distance(self.pts[4], self.pts[11]) > 50) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]):
                ch1 = 4
                # print("44444")

        # con for [l][gh]
        l = [[3, 6], [3, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[4][0] < self.pts[0][0]):
                ch1 = 4
                # print("44444")

        # con for [l][c0]
        l = [[2, 2], [2, 5], [2, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[1][0] < self.pts[12][0]):
                ch1 = 4
                # print("44444")

        # con for [l][c0]
        l = [[2, 2], [2, 5], [2, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[1][0] < self.pts[12][0]):
                ch1 = 4
                # print("44444")

        # con for [gh][z]
        l = [[3, 6], [3, 5], [3, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][
                1]) and self.pts[4][1] > self.pts[10][1]:
                ch1 = 5
                print("55555b")

        # con for [gh][pq]
        l = [[3, 2], [3, 1], [3, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[4][1] + 17 > self.pts[8][1] and self.pts[4][1] + 17 > self.pts[12][1] and self.pts[4][1] + 17 > self.pts[16][1] and self.pts[4][
                1] + 17 > self.pts[20][1]:
                ch1 = 5
                print("55555a")

        # con for [l][pqz]
        l = [[4, 4], [4, 5], [4, 2], [7, 5], [7, 6], [7, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[4][0] > self.pts[0][0]:
                ch1 = 5
                # print("55555")

        # con for [pqz][aemnst]
        l = [[0, 2], [0, 6], [0, 1], [0, 5], [0, 0], [0, 7], [0, 4], [0, 3], [2, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[0][0] < self.pts[8][0] and self.pts[0][0] < self.pts[12][0] and self.pts[0][0] < self.pts[16][0] and self.pts[0][0] < self.pts[20][0]:
                ch1 = 5
                # print("55555")

        # con for [pqz][yj]
        l = [[5, 7], [5, 2], [5, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[3][0] < self.pts[0][0]:
                ch1 = 7
                # print("77777")

        # con for [l][yj]
        l = [[4, 6], [4, 2], [4, 4], [4, 1], [4, 5], [4, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[6][1] < self.pts[8][1]:
                ch1 = 7
                # print("77777")

        # con for [x][yj]
        l = [[6, 7], [0, 7], [0, 1], [0, 0], [6, 4], [6, 6], [6, 5], [6, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[18][1] > self.pts[20][1]:
                ch1 = 7
                # print("77777")

        # condition for [x][aemnst]
        l = [[0, 4], [0, 2], [0, 3], [0, 1], [0, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[5][0] > self.pts[16][0]:
                ch1 = 6
                print("666661")


        # condition for [yj][x]
        print("2222  ch1=+++++++++++++++++", ch1, ",", ch2)
        l = [[7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[18][1] < self.pts[20][1] and self.pts[8][1] < self.pts[10][1]:
                ch1 = 6
                print("666662")

        # condition for [c0][x]
        l = [[2, 1], [2, 2], [2, 6], [2, 7], [2, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[8], self.pts[16]) > 50:
                ch1 = 6
                print("666663")

        # con for [l][x]

        l = [[4, 6], [4, 2], [4, 1], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[4], self.pts[11]) < 60:
                ch1 = 6
                print("666664")

        # con for [x][d]
        l = [[1, 4], [1, 6], [1, 0], [1, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[5][0] - self.pts[4][0] - 15 > 0:
                ch1 = 6
                print("666665")

        # con for [b][pqz]
        l = [[5, 0], [5, 1], [5, 4], [5, 5], [5, 6], [6, 1], [7, 6], [0, 2], [7, 1], [7, 4], [6, 6], [7, 2], [5, 0],
             [6, 3], [6, 4], [7, 5], [7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][
                1]):
                ch1 = 1
                print("111111")

        # con for [f][pqz]
        l = [[6, 1], [6, 0], [0, 3], [6, 4], [2, 2], [0, 6], [6, 2], [7, 6], [4, 6], [4, 1], [4, 2], [0, 2], [7, 1],
             [7, 4], [6, 6], [7, 2], [7, 5], [7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and
                    self.pts[18][1] > self.pts[20][1]):
                ch1 = 1
                print("111112")

        l = [[6, 1], [6, 0], [4, 2], [4, 1], [4, 6], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and
                    self.pts[18][1] > self.pts[20][1]):
                ch1 = 1
                print("111112")

        # con for [d][pqz]
        fg = 19
        # print("_________________ch1=",ch1," ch2=",ch2)
        l = [[5, 0], [3, 4], [3, 0], [3, 1], [3, 5], [5, 5], [5, 4], [5, 1], [7, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
                 self.pts[18][1] < self.pts[20][1]) and (self.pts[2][0] < self.pts[0][0]) and self.pts[4][1] > self.pts[14][1]):
                ch1 = 1
                print("111113")

        l = [[4, 1], [4, 2], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.distance(self.pts[4], self.pts[11]) < 50) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]):
                ch1 = 1
                print("1111993")

        l = [[3, 4], [3, 0], [3, 1], [3, 5], [3, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
                 self.pts[18][1] < self.pts[20][1]) and (self.pts[2][0] < self.pts[0][0]) and self.pts[14][1] < self.pts[4][1]):
                ch1 = 1
                print("1111mmm3")

        l = [[6, 6], [6, 4], [6, 1], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[5][0] - self.pts[4][0] - 15 < 0:
                ch1 = 1
                print("1111140")

        # con for [i][pqz]
        l = [[5, 4], [5, 5], [5, 1], [0, 3], [0, 7], [5, 0], [0, 2], [6, 2], [7, 5], [7, 1], [7, 6], [7, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
                 self.pts[18][1] > self.pts[20][1])):
                ch1 = 1
                print("111114")

        # con for [yj][bfdi]
        l = [[1, 5], [1, 7], [1, 1], [1, 6], [1, 3], [1, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[4][0] < self.pts[5][0] + 15) and (
            (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
             self.pts[18][1] > self.pts[20][1])):
                ch1 = 7
                print("111114lll;;p")

        # con for [uvr]
        l = [[5, 5], [5, 0], [5, 4], [5, 1], [4, 6], [4, 1], [7, 6], [3, 0], [3, 5]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
                 self.pts[18][1] < self.pts[20][1])) and self.pts[4][1] > self.pts[14][1]:
                ch1 = 1
                print("111115")

        # con for [w]
        fg = 13
        l = [[3, 5], [3, 0], [3, 6], [5, 1], [4, 1], [2, 0], [5, 0], [5, 5]]
        pl = [ch1, ch2]
        if pl in l:
            if not (self.pts[0][0] + fg < self.pts[8][0] and self.pts[0][0] + fg < self.pts[12][0] and self.pts[0][0] + fg < self.pts[16][0] and
                    self.pts[0][0] + fg < self.pts[20][0]) and not (
                    self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][0] and self.pts[0][0] > self.pts[20][
                0]) and self.distance(self.pts[4], self.pts[11]) < 50:
                ch1 = 1
                print("111116")

        # con for [w]

        l = [[5, 0], [5, 5], [0, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1]:
                ch1 = 1
                print("1117")

        # -------------------------condn for 8 groups  ends

        # -------------------------condn for subgroups  starts
        #
        if ch1 == 0:
            ch1 = 'S'
            if self.pts[4][0] < self.pts[6][0] and self.pts[4][0] < self.pts[10][0] and self.pts[4][0] < self.pts[14][0] and self.pts[4][0] < self.pts[18][0]:
                ch1 = 'A'
            if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] < self.pts[10][0] and self.pts[4][0] < self.pts[14][0] and self.pts[4][0] < self.pts[18][
                0] and self.pts[4][1] < self.pts[14][1] and self.pts[4][1] < self.pts[18][1]:
                ch1 = 'T'
            if self.pts[4][1] > self.pts[8][1] and self.pts[4][1] > self.pts[12][1] and self.pts[4][1] > self.pts[16][1] and self.pts[4][1] > self.pts[20][1]:
                ch1 = 'E'
            if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] > self.pts[10][0] and self.pts[4][0] > self.pts[14][0] and self.pts[4][1] < self.pts[18][1]:
                ch1 = 'M'
            if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] > self.pts[10][0] and self.pts[4][1] < self.pts[18][1] and self.pts[4][1] < self.pts[14][1]:
                ch1 = 'N'

        if ch1 == 2:
            if self.distance(self.pts[12], self.pts[4]) > 42:
                ch1 = 'C'
            else:
                ch1 = 'O'

        if ch1 == 3:
            if (self.distance(self.pts[8], self.pts[12])) > 72:
                ch1 = 'G'
            else:
                ch1 = 'H'

        if ch1 == 7:
            if self.distance(self.pts[8], self.pts[4]) > 42:
                ch1 = 'Y'
            else:
                ch1 = 'J'

        if ch1 == 4:
            ch1 = 'L'

        if ch1 == 6:
            ch1 = 'X'

        if ch1 == 5:
            if self.pts[4][0] > self.pts[12][0] and self.pts[4][0] > self.pts[16][0] and self.pts[4][0] > self.pts[20][0]:
                if self.pts[8][1] < self.pts[5][1]:
                    ch1 = 'Z'
                else:
                    ch1 = 'Q'
            else:
                ch1 = 'P'

        if ch1 == 1:
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][
                1]):
                ch1 = 'B'
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][
                1]):
                ch1 = 'D'
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][
                1]):
                ch1 = 'F'
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] > self.pts[20][
                1]):
                ch1 = 'I'
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] < self.pts[20][
                1]):
                ch1 = 'W'
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][
                1]) and self.pts[4][1] < self.pts[9][1]:
                ch1 = 'K'
            if ((self.distance(self.pts[8], self.pts[12]) - self.distance(self.pts[6], self.pts[10])) < 8) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]):
                ch1 = 'U'
            if ((self.distance(self.pts[8], self.pts[12]) - self.distance(self.pts[6], self.pts[10])) >= 8) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]) and (self.pts[4][1] > self.pts[9][1]):
                ch1 = 'V'

            if (self.pts[8][0] > self.pts[12][0]) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]):
                ch1 = 'R'

        if ch1 == 1 or ch1 =='E' or ch1 =='S' or ch1 =='X' or ch1 =='Y' or ch1 =='B':
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1=" "



        print(self.pts[4][0] < self.pts[5][0])
        if ch1 == 'E' or ch1=='Y' or ch1=='B':
            if (self.pts[4][0] < self.pts[5][0]) and (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1="next"

        # Fixed classic Python logical bug: using in list check
        if ch1 in ['next', 'Next', 'B', 'C', 'H', 'F', 'X']:
            if (self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][0] and self.pts[0][0] > self.pts[20][0]) and (self.pts[4][1] < self.pts[8][1] and self.pts[4][1] < self.pts[12][1] and self.pts[4][1] < self.pts[16][1] and self.pts[4][1] < self.pts[20][1]) and (self.pts[4][1] < self.pts[6][1] and self.pts[4][1] < self.pts[10][1] and self.pts[4][1] < self.pts[14][1] and self.pts[4][1] < self.pts[18][1]):
                ch1 = 'Backspace'

        # Set symbol output for prediction
        self.current_symbol = ch1


    def destructor(self):

        print("Closing Application...")
        print(self.ten_prev_char)
        if self.tts_available:
            try:
                self.engine.endLoop()
            except Exception:
                pass
        self.root.destroy()
        self.vs.release()
        cv2.destroyAllWindows()


print("Starting Application...")

(Application()).root.mainloop()
