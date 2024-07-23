import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from enum import Enum

class Color(Enum):
    BLUE = 0
    GREEN = 1
    RED = 2
    YELLOW = 3

# Giving different arrays to handle colour points of different colour
points = {Color.BLUE: [deque(maxlen=1024)], Color.GREEN: [deque(maxlen=1024)], Color.RED: [deque(maxlen=1024)], Color.YELLOW: [deque(maxlen=1024)]}

# These indexes will be used to mark the points in particular arrays of specific color
index = {Color.BLUE: 0, Color.GREEN: 0, Color.RED: 0, Color.YELLOW: 0}

# The kernel to be used for dilation purpose 
kernel = np.ones((5, 5), np.uint8)

colors = {Color.BLUE: (255, 0, 0), Color.GREEN: (0, 255, 0), Color.RED: (0, 0, 255), Color.YELLOW: (0, 255, 255)}
selected_colors = []

# Canvas setup
def setup_canvas():
    canvas = np.zeros((471, 636, 3)) + 255
    buttons = [("CLEAR", (40, 1), (140, 65)), 
               ("BLUE", (160, 1), (255, 65)), 
               ("GREEN", (275, 1), (370, 65)), 
               ("RED", (390, 1), (485, 65)), 
               ("YELLOW", (505, 1), (600, 65))]
    for text, start, end in buttons:
        canvas = cv2.rectangle(canvas, start, end, (0, 0, 0), 2)
        cv2.putText(canvas, text, (start[0]+10, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    return canvas

paintWindow = setup_canvas()
cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)

def draw_buttons(frame):
    buttons = [("CLEAR", (40, 1), (140, 65)), 
               ("BLUE", (160, 1), (255, 65)), 
               ("GREEN", (275, 1), (370, 65)), 
               ("RED", (390, 1), (485, 65)), 
               ("YELLOW", (505, 1), (600, 65))]
    for text, start, end in buttons:
        frame = cv2.rectangle(frame, start, end, (0, 0, 0), 2)
        cv2.putText(frame, text, (start[0]+10, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    return frame

def highlight_selected_colors(frame):
    button_positions = {Color.BLUE: (160, 1, 255, 65), Color.GREEN: (275, 1, 370, 65), Color.RED: (390, 1, 485, 65), Color.YELLOW: (505, 1, 600, 65)}
    for color in selected_colors:
        x1, y1, x2, y2 = button_positions[color]
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), colors[color], 2)
    return frame

def blend_colors(color1, color2):
    blended_color = tuple(map(lambda x: int((x[0] + x[1]) / 2), zip(color1, color2)))
    return blended_color

def handle_landmarks(landmarks, frame):
    global selected_colors
    fore_finger = (landmarks[8][0], landmarks[8][1])
    center = fore_finger
    thumb = (landmarks[4][0], landmarks[4][1])
    cv2.circle(frame, center, 3, (0, 255, 0), -1)
    if thumb[1] - center[1] < 30:
        for color in Color:
            points[color].append(deque(maxlen=512))
            index[color] += 1
    elif center[1] <= 65:
        if 40 <= center[0] <= 140:
            for color in Color:
                points[color] = [deque(maxlen=512)]
                index[color] = 0
            paintWindow[67:, :, :] = 255
            selected_colors = []
        elif 160 <= center[0] <= 255:
            if Color.BLUE not in selected_colors:
                selected_colors.append(Color.BLUE)
        elif 275 <= center[0] <= 370:
            if Color.GREEN not in selected_colors:
                selected_colors.append(Color.GREEN)
        elif 390 <= center[0] <= 485:
            if Color.RED not in selected_colors:
                selected_colors.append(Color.RED)
        elif 505 <= center[0] <= 600:
            if Color.YELLOW not in selected_colors:
                selected_colors.append(Color.YELLOW)
        if len(selected_colors) > 2:
            selected_colors = selected_colors[-2:]
    else:
        if len(selected_colors) == 1:
            points[selected_colors[0]][index[selected_colors[0]]].appendleft(center)
        elif len(selected_colors) == 2:
            blended_color = blend_colors(colors[selected_colors[0]], colors[selected_colors[1]])
            cv2.circle(frame, center, 3, blended_color, -1)
            cv2.circle(paintWindow, center, 3, blended_color, -1)
            points[selected_colors[0]][index[selected_colors[0]]].appendleft(center)
            points[selected_colors[1]][index[selected_colors[1]]].appendleft(center)

def draw_lines(frame):
    for color in Color:
        for i in range(len(points[color])):
            for j in range(1, len(points[color][i])):
                if points[color][i][j - 1] is None or points[color][i][j] is None:
                    continue
                if len(selected_colors) == 2:
                    blended_color = blend_colors(colors[selected_colors[0]], colors[selected_colors[1]])
                else:
                    blended_color = colors[color]
                cv2.line(frame, points[color][i][j - 1], points[color][i][j], blended_color, 2)
                cv2.line(paintWindow, points[color][i][j - 1], points[color][i][j], blended_color, 2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = draw_buttons(frame)
    frame = highlight_selected_colors(frame)

    result = hands.process(framergb)
    if result.multi_hand_landmarks:
        landmarks = [[int(lm.x * 640), int(lm.y * 480)] for lm in result.multi_hand_landmarks[0].landmark]
        mpDraw.draw_landmarks(frame, result.multi_hand_landmarks[0], mpHands.HAND_CONNECTIONS)
        handle_landmarks(landmarks, frame)

    draw_lines(frame)

    cv2.imshow("Output", frame)
    cv2.imshow("Paint", paintWindow)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
