import cv2
import mediapipe as mp
import numpy as np

mp_draw = mp.solutions.drawing_utils
mp_hand = mp.solutions.hands
video = cv2.VideoCapture(0)

width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
canvas = np.ones((height, width, 3), dtype="uint8") * 255

prev_x, prev_y = None, None
lines = []

with mp_hand.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while True:
        ret, frame = video.read()
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            hand_landmark = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand_landmark, mp_hand.HAND_CONNECTIONS)

            landmark_7_vertical = int(hand_landmark.landmark[7].y * frame.shape[0])
            landmark_8_horizontal = int(hand_landmark.landmark[8].x * frame.shape[1])
            landmark_8_vertical = int(hand_landmark.landmark[8].y * frame.shape[0])

            landmark_4_vertical = int(hand_landmark.landmark[4].y * frame.shape[0])
            landmark_11_vertical = int(hand_landmark.landmark[11].y * frame.shape[0])
            landmark_12_vertical = int(hand_landmark.landmark[12].y * frame.shape[0])
            landmark_15_vertical = int(hand_landmark.landmark[15].y * frame.shape[0])
            landmark_16_vertical = int(hand_landmark.landmark[16].y * frame.shape[0])
            landmark_19_vertical = int(hand_landmark.landmark[19].y * frame.shape[0])
            landmark_20_vertical = int(hand_landmark.landmark[20].y * frame.shape[0])

            index_finger_extended = (
                landmark_4_vertical < landmark_12_vertical and
                landmark_8_vertical < landmark_11_vertical
            )
            fingers_extended = (
                landmark_8_vertical < landmark_7_vertical and
                landmark_12_vertical < landmark_11_vertical and
                landmark_16_vertical < landmark_15_vertical and
                landmark_20_vertical < landmark_19_vertical
            )

            if index_finger_extended:
                if prev_x is not None and prev_y is not None:
                    cv2.line(canvas, (prev_x, prev_y), (landmark_8_horizontal, landmark_8_vertical), (0, 0, 0), 5)
                    lines.append(((prev_x, prev_y), (landmark_8_horizontal, landmark_8_vertical)))
                prev_x, prev_y = landmark_8_horizontal, landmark_8_vertical
            else:
                prev_x, prev_y = None, None

            if fingers_extended:
                if lines:
                    lines.pop()
                    canvas = np.ones((height, width, 3), dtype="uint8") * 255
                    for line in lines:
                        cv2.line(canvas, line[0], line[1], (0, 0, 0), 5)

            cv2.circle(frame, (landmark_8_horizontal, landmark_8_vertical), 5, (200, 200, 200), -1)

        else:
            prev_x, prev_y = None, None

        cv2.imshow("Canvas", canvas)
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

video.release()
cv2.destroyAllWindows()
