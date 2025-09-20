# hand_control.py
import time
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
from pynput.mouse import Controller as MouseController, Button
from pynput.keyboard import Controller as KeyboardController, Key

# === CONFIG ===
CAM_INDEX = 1
DETECTION_CONFIDENCE = 0.6
TRACKING_CONFIDENCE = 0.6

LEFT_CLICK_COOLDOWN = 0.5   # seconds between left clicks
RIGHT_CLICK_COOLDOWN = 0.5  # seconds between right clicks
ALT_TAB_COOLDOWN = 1.2      # seconds between alt-tab triggers
SWIPE_HISTORY = 6           # frames to check for horizontal movement
SWIPE_THRESHOLD = 0.12      # normalized x movement to consider a swipe

SHOW_OVERLAY = True

# === SETUP ===
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

mouse = MouseController()
keyboard = KeyboardController()

# Keep last trigger times
last_left_click = 0.0
last_right_click = 0.0
last_alt_tab = 0.0

# For swipe detection
wrist_history = deque(maxlen=SWIPE_HISTORY)

# Helper: determine which fingers are up
def fingers_up(hand_landmarks, hand_label='Right'):
    # indices: 4 thumb tip, 8 index tip, 12 middle tip, 16 ring tip, 20 pinky tip
    # We'll compare tip landmark with pip (lower joint): for finger i, use tip vs pip landmarks
    lm = hand_landmarks.landmark
    fingers = {}
    # Note: Mediapipe uses normalized coordinates (x,y) where y increases downwards.

    # Thumb: compare x of tip and ip depending on left/right
    # For simplicity: thumb up if tip.x is to the right of ip.x for right hand (hand_label)
    if hand_label == 'Right':
        fingers['thumb'] = lm[4].x > lm[3].x
    else:
        fingers['thumb'] = lm[4].x < lm[3].x

    # Other fingers: tip.y < pip.y  -> finger extended (because y grows downward)
    fingers['index'] = lm[8].y < lm[6].y
    fingers['middle'] = lm[12].y < lm[10].y
    fingers['ring'] = lm[16].y < lm[14].y
    fingers['pinky'] = lm[20].y < lm[18].y

    return fingers

# Map normalized coordinates to screen coordinates (approx)
def normalized_to_screen(norm_x, norm_y, frame_w, frame_h):
    # webcam frame origin top-left, normalized x in [0,1]
    px = int(norm_x * frame_w)
    py = int(norm_y * frame_h)
    return px, py

# Trigger OS actions
def do_left_click(x=None, y=None):
    global last_left_click
    now = time.time()
    if now - last_left_click < LEFT_CLICK_COOLDOWN:
        return False
    last_left_click = now
    # Move mouse if coordinates provided (they should be in screen coordinates)
    if x is not None and y is not None:
        try:
            mouse.position = (x, y)
        except Exception:
            pass
    mouse.click(Button.left, 1)
    return True

def do_right_click(x=None, y=None):
    global last_right_click
    now = time.time()
    if now - last_right_click < RIGHT_CLICK_COOLDOWN:
        return False
    last_right_click = now
    if x is not None and y is not None:
        try:
            mouse.position = (x, y)
        except Exception:
            pass
    mouse.click(Button.right, 1)
    return True

def do_alt_tab():
    global last_alt_tab
    now = time.time()
    if now - last_alt_tab < ALT_TAB_COOLDOWN:
        return False
    last_alt_tab = now
    # Press Alt+Tab once to go to next app
    # Hold alt, press tab, release alt
    keyboard.press(Key.alt)
    keyboard.press(Key.tab)
    keyboard.release(Key.tab)
    time.sleep(0.05)
    keyboard.release(Key.alt)
    return True

# Main
def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=DETECTION_CONFIDENCE,
        min_tracking_confidence=TRACKING_CONFIDENCE
    ) as hands:
        print("Starting — press ESC to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            frame = cv2.flip(frame, 1)  # mirror image (natural)
            frame_h, frame_w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            action_text = ""
            if results.multi_hand_landmarks:
                # use first detected hand
                hand_landmarks = results.multi_hand_landmarks[0]
                hand_label = 'Right'
                # Mediapipe hands solution doesn't directly give left/right with Hands default;
                # If multi_handedness available, use it:
                if results.multi_handedness:
                    try:
                        hand_label = results.multi_handedness[0].classification[0].label
                    except Exception:
                        hand_label = 'Right'

                # landmarks
                lm = hand_landmarks.landmark
                # wrist normalized coords
                wrist_x = lm[0].x
                wrist_y = lm[0].y
                wrist_history.append(wrist_x)

                fup = fingers_up(hand_landmarks, hand_label)

                # Compute fingertip position (index tip) in screen coords
                idx_x_norm = lm[8].x
                idx_y_norm = lm[8].y
                idx_px, idx_py = normalized_to_screen(idx_x_norm, idx_y_norm, frame_w, frame_h)

                # Gesture rules (tunable)
                # 1) Index up only => left click at index tip
                if fup['index'] and not fup['middle'] and not fup['ring'] and not fup['pinky']:
                    clicked = do_left_click(idx_px, idx_py)
                    if clicked:
                        action_text = "LEFT CLICK"
                # 2) Index + middle => right click
                elif fup['index'] and fup['middle'] and not fup['ring'] and not fup['pinky']:
                    clicked = do_right_click(idx_px, idx_py)
                    if clicked:
                        action_text = "RIGHT CLICK"
                # 3) Index+middle+ring up -> check for horizontal swipe to trigger alt-tab
                elif fup['index'] and fup['middle'] and fup['ring']:
                    # need enough history
                    if len(wrist_history) >= SWIPE_HISTORY:
                        dx = wrist_history[-1] - wrist_history[0]
                        if abs(dx) > SWIPE_THRESHOLD:
                            triggered = do_alt_tab()
                            if triggered:
                                action_text = "ALT+TAB"
                                wrist_history.clear()  # prevent repeated triggers

                # draw landmarks
                if SHOW_OVERLAY:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # draw small dot where index tip is
                    cv2.circle(frame, (idx_px, idx_py), 8, (0,255,0), -1)

                    # write finger states
                    txt = f"Thumb:{int(fup['thumb'])} Index:{int(fup['index'])} Mid:{int(fup['middle'])} Ring:{int(fup['ring'])} Pinky:{int(fup['pinky'])}"
                    cv2.putText(frame, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,0), 2)

            else:
                wrist_history.clear()

            if SHOW_OVERLAY and action_text:
                cv2.putText(frame, action_text, (10, frame_h - 20), cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 200, 0), 3)

            cv2.imshow("Hand Control — ESC to quit", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
