import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import os, os.path
from keras.models import load_model
import traceback

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "AtoZ_3.1")
WHITE_IMG_PATH = os.path.join(BASE_DIR, "white.jpg")

capture = cv2.VideoCapture(0)

hd = HandDetector(maxHands=1)
hd2 = HandDetector(maxHands=1)

p_dir = "A"
c_dir = "a"

target_folder = os.path.join(DATASET_DIR, p_dir)
os.makedirs(target_folder, exist_ok=True)
count = len(os.listdir(target_folder))

offset = 30
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
        img_final = img_final1 = img_final2 = 0

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            image = frame[max(0, y - offset):min(frame.shape[0], y + h + offset),
                          max(0, x - offset):min(frame.shape[1], x + w + offset)]

            if image.size > 0:
                roi = image
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (1, 1), 2)

                gray2 = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                blur2 = cv2.GaussianBlur(gray2, (5, 5), 2)
                th3 = cv2.adaptiveThreshold(blur2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
                ret, test_image = cv2.threshold(th3, 27, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                test_image1 = blur
                img_final1 = np.ones((400, 400), np.uint8) * 148
                h_img = test_image1.shape[0]
                w_img = test_image1.shape[1]
                img_final1[((400 - h_img) // 2):((400 - h_img) // 2) + h_img, ((400 - w_img) // 2):((400 - w_img) // 2) + w_img] = test_image1

                img_final = np.ones((400, 400), np.uint8) * 255
                h_img = test_image.shape[0]
                w_img = test_image.shape[1]
                img_final[((400 - h_img) // 2):((400 - h_img) // 2) + h_img, ((400 - w_img) // 2):((400 - w_img) // 2) + w_img] = test_image

        hands = hd.findHands(frame, draw=False, flipType=True)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            image = frame[max(0, y - offset):min(frame.shape[0], y + h + offset),
                          max(0, x - offset):min(frame.shape[1], x + w + offset)]
            white = cv2.imread(WHITE_IMG_PATH)
            if white is None:
                white = np.ones((400, 400, 3), np.uint8) * 255

            if image.size > 0:
                handz = hd2.findHands(image, draw=False, flipType=True)
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

                    cv2.imshow("skeleton", white)

            if isinstance(img_final, np.ndarray):
                cv2.imshow("binary", img_final)

        cv2.imshow("frame", frame)
        interrupt = cv2.waitKey(1)
        if interrupt & 0xFF == 27:
            break
        if interrupt & 0xFF == ord('n'):
            p_dir = chr(ord(p_dir) + 1)
            c_dir = chr(ord(c_dir) + 1)
            if ord(p_dir) == ord('Z') + 1:
                p_dir = "A"
                c_dir = "a"
            flag = False
            target_folder = os.path.join(DATASET_DIR, p_dir)
            os.makedirs(target_folder, exist_ok=True)
            count = len(os.listdir(target_folder))

        if interrupt & 0xFF == ord('a'):
            flag = not flag
            if flag:
                suv = 0

        if flag:
            if suv == 50:
                flag = False
            if step % 2 == 0:
                target_folder = os.path.join(DATASET_DIR, p_dir)
                os.makedirs(target_folder, exist_ok=True)
                if isinstance(img_final1, np.ndarray):
                    save_path = os.path.join(target_folder, f"{c_dir}{count}.jpg")
                    cv2.imwrite(save_path, img_final1)
                count += 1
                suv += 1
            step += 1
    except Exception:
        print("==", traceback.format_exc())

capture.release()
cv2.destroyAllWindows()























































# img_final=cv2.resize(img_final,(224,224));
# img_finalf=np.ones((400,400,3),np.uint8)*255;
# print("img final shape= ", img_final)
#  for i in range(400):
#      for j in range(400):
#          if(img_final[i][j]==255):
#              img_finalf[i][j]=[255,255,255]
#          else:
#              img_finalf[i][j]=[0,0,0];
# print("img final f shape= ", img_finalf)
# image = cv2.medianBlur(test_image, 5)
# kernel = np.ones((3, 3), np.uint8)
# kernel1 = np.ones((1, 1), np.uint8)
# dilate = cv2.dilate(image, kernel, iterations=1)
# dilate = cv2.erode(dilate, kernel1, iterations=1)

# cv2.imshow("gray",gray)
# cv2.imshow("blurr",blur)
# cv2.imshow("adapt threshold",th3)
# cv2.imshow("roi",test_image)

# white increase

# if flag:
#     if step % 2 == 0:
#         cv2.imwrite("D:\\sign_data\\B\\b" + str(count) + ".jpg", img_final)
#         print(count)
#         count += 1
#     step += 1