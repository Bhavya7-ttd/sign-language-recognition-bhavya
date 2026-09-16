import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import os
import traceback

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "AtoZ_3.1")
WHITE_IMG_PATH = os.path.join(BASE_DIR, "white.jpg")

capture = cv2.VideoCapture(0)
hd = HandDetector(maxHands=1)
hd2 = HandDetector(maxHands=1)

c_dir = 'A'
target_folder = os.path.join(DATASET_DIR, c_dir)
os.makedirs(target_folder, exist_ok=True)
count = len(os.listdir(target_folder))

offset = 15
step = 1
flag = False
suv = 0

white = np.ones((400, 400, 3), np.uint8) * 255
cv2.imwrite(WHITE_IMG_PATH, white)

while True:
    try:
        _, frame = capture.read()
        if frame is None:
            continue
        frame = cv2.flip(frame, 1)
        hands = hd.findHands(frame, draw=False, flipType=True)
        white = cv2.imread(WHITE_IMG_PATH)
        if white is None:
            white = np.ones((400, 400, 3), np.uint8) * 255

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            image = np.array(frame[max(0, y - offset):min(frame.shape[0], y + h + offset),
                                  max(0, x - offset):min(frame.shape[1], x + w + offset)])

            if image.size > 0:
                handz, imz = hd2.findHands(image, draw=True, flipType=True)
                if handz:
                    hand = handz[0]
                    pts = hand['lmList']
                    os_val = ((400 - w) // 2) - 15
                    os1_val = ((400 - h) // 2) - 15
                    for t in range(0, 4, 1):
                        cv2.line(white, (pts[t][0] + os_val, pts[t][1] + os1_val), (pts[t + 1][0] + os_val, pts[t + 1][1] + os1_val), (0, 255, 0), 3)
                    for t in range(5, 8, 1):
                        cv2.line(white, (pts[t][0] + os_val, pts[t][1] + os1_val), (pts[t + 1][0] + os_val, pts[t + 1][1] + os1_val), (0, 255, 0), 3)
                    for t in range(9, 12, 1):
                        cv2.line(white, (pts[t][0] + os_val, pts[t][1] + os1_val), (pts[t + 1][0] + os_val, pts[t + 1][1] + os1_val), (0, 255, 0), 3)
                    for t in range(13, 16, 1):
                        cv2.line(white, (pts[t][0] + os_val, pts[t][1] + os1_val), (pts[t + 1][0] + os_val, pts[t + 1][1] + os1_val), (0, 255, 0), 3)
                    for t in range(17, 20, 1):
                        cv2.line(white, (pts[t][0] + os_val, pts[t][1] + os1_val), (pts[t + 1][0] + os_val, pts[t + 1][1] + os1_val), (0, 255, 0), 3)
                    cv2.line(white, (pts[5][0] + os_val, pts[5][1] + os1_val), (pts[9][0] + os_val, pts[9][1] + os1_val), (0, 255, 0), 3)
                    cv2.line(white, (pts[9][0] + os_val, pts[9][1] + os1_val), (pts[13][0] + os_val, pts[13][1] + os1_val), (0, 255, 0), 3)
                    cv2.line(white, (pts[13][0] + os_val, pts[13][1] + os1_val), (pts[17][0] + os_val, pts[17][1] + os1_val), (0, 255, 0), 3)
                    cv2.line(white, (pts[0][0] + os_val, pts[0][1] + os1_val), (pts[5][0] + os_val, pts[5][1] + os1_val), (0, 255, 0), 3)
                    cv2.line(white, (pts[0][0] + os_val, pts[0][1] + os1_val), (pts[17][0] + os_val, pts[17][1] + os1_val), (0, 255, 0), 3)

                    for i in range(21):
                        cv2.circle(white, (pts[i][0] + os_val, pts[i][1] + os1_val), 2, (0, 0, 255), 1)

                    skeleton1 = np.array(white)
                    cv2.imshow("1", skeleton1)

        frame = cv2.putText(frame, "dir=" + str(c_dir) + "  count=" + str(count), (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow("frame", frame)
        interrupt = cv2.waitKey(1)
        if interrupt & 0xFF == 27:
            break

        if interrupt & 0xFF == ord('n'):
            c_dir = chr(ord(c_dir) + 1)
            if ord(c_dir) == ord('Z') + 1:
                c_dir = 'A'
            flag = False
            target_folder = os.path.join(DATASET_DIR, c_dir)
            os.makedirs(target_folder, exist_ok=True)
            count = len(os.listdir(target_folder))

        if interrupt & 0xFF == ord('a'):
            flag = not flag
            if flag:
                suv = 0

        if flag:
            if suv == 180:
                flag = False
            if step % 3 == 0:
                save_path = os.path.join(target_folder, f"{count}.jpg")
                cv2.imwrite(save_path, skeleton1)
                count += 1
                suv += 1
            step += 1

    except Exception:
        print("==", traceback.format_exc())

capture.release()
cv2.destroyAllWindows()