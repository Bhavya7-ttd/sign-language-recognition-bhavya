import cv2
from cvzone.HandTrackingModule import HandDetector

# Hand detector
detector = HandDetector(
    staticMode=False,
    maxHands=1,
    modelComplexity=1,
    detectionCon=0.5,
    minTrackCon=0.5
)

cap = cv2.VideoCapture(0)

# Set camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    success, frame = cap.read()

    if not success:
        print("Could not read camera")
        break

    # Mirror image
    frame = cv2.flip(frame, 1)

    # Detect hand
    hands, frame = detector.findHands(
        frame,
        draw=True,
        flipType=False
    )

    if hands:
        print("HAND DETECTED")

        hand = hands[0]

        print("Bounding box:", hand["bbox"])
        print("Landmarks:", len(hand["lmList"]))

        # Show bbox information
        x, y, w, h = hand["bbox"]

        cv2.putText(
            frame,
            "HAND DETECTED",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    else:
        print("No hand detected")

        cv2.putText(
            frame,
            "NO HAND",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    cv2.imshow("Hand Detection Test", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()