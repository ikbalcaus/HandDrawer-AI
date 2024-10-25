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

with mp_hand.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while True:
        ret, frame = video.read()

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmark, mp_hand.HAND_CONNECTIONS)

                x_index = int(hand_landmark.landmark[8].x * frame.shape[1])
                y_index = int(hand_landmark.landmark[8].y * frame.shape[0])

                if prev_x is not None and prev_y is not None:
                    cv2.line(canvas, (prev_x, prev_y), (x_index, y_index), (0, 0, 0), 5)

                prev_x, prev_y = x_index, y_index

        else:
            prev_x, prev_y = None, None

        cv2.imshow("Drawing Canvas", canvas)
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

video.release()
cv2.destroyAllWindows()
