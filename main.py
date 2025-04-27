import cv2
import time
import math
import numpy as np
import pyautogui
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Import HandTrackingModule
import HandTrackingModule as htm

# Camera settings
wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)  # Use 0 for default camera
cap.set(3, wCam)
cap.set(4, hCam)

# Initialize Hand Detector
detector = htm.handDetector(maxHands=1, detectionCon=0.85, trackCon=0.8)

# System Volume Control using Pycaw
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()  # (-63.5 dB, 0 dB)

minVol = -63
maxVol = volRange[1]
hmin = 50
hmax = 200
volBar = 400
volPer = 0
vol = 0
color = (0, 215, 255)

# Finger tip IDs
tipIds = [4, 8, 12, 16, 20]

mode = ''
active = 0
pyautogui.FAILSAFE = False

pTime = 0  # Previous Time for FPS calculation

# ✅ **Move `putText` function to the top**
def putText(img, text, loc=(250, 450), color=(0, 255, 255)):
    """Displays text on the screen."""
    cv2.putText(img, str(text), loc, cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, color, 3)

while True:
    success, img = cap.read()
    if not success:
        continue

    img = cv2.flip(img, 1)  # ✅ Fix camera flipping (mirror effect)
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    fingers = []
    if len(lmList) != 0:
        # Thumb (special case)
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Remaining fingers
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:  # Finger is up
                fingers.append(1)
            else:
                fingers.append(0)

        # Mode Selection
        if fingers == [0, 0, 0, 0, 0] and active == 0:
            mode = 'N'
        elif (fingers == [0, 1, 0, 0, 0] or fingers == [0, 1, 1, 0, 0]) and active == 0:
            mode = 'Scroll'
            active = 1
        elif fingers == [1, 1, 0, 0, 0] and active == 0:
            mode = 'Volume'
            active = 1
        elif fingers == [1, 1, 1, 1, 1] and active == 0:
            mode = 'Cursor'
            active = 1

    # SCROLLING MODE
    if mode == 'Scroll':
        active = 1
        putText(img, mode)
        cv2.rectangle(img, (200, 410), (245, 460), (255, 255, 255), cv2.FILLED)

        if len(lmList) != 0:
            if fingers == [0, 1, 0, 0, 0]:  # Scroll Up
                putText(img, 'U', (200, 455), (0, 255, 0))
                pyautogui.scroll(300)
            if fingers == [0, 1, 1, 0, 0]:  # Scroll Down
                putText(img, 'D', (200, 455), (0, 0, 255))
                pyautogui.scroll(-300)
            elif fingers == [0, 0, 0, 0, 0]:
                active = 0
                mode = 'N'

    # VOLUME CONTROL MODE
    if mode == 'Volume':
        active = 1
        putText(img, mode)

        if len(lmList) != 0:
            if fingers[-1] == 1:
                active = 0
                mode = 'N'
            else:
                x1, y1 = lmList[4][1], lmList[4][2]  # Thumb tip
                x2, y2 = lmList[8][1], lmList[8][2]  # Index finger tip
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                cv2.circle(img, (x1, y1), 10, color, cv2.FILLED)
                cv2.circle(img, (x2, y2), 10, color, cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), color, 3)
                cv2.circle(img, (cx, cy), 8, color, cv2.FILLED)

                length = math.hypot(x2 - x1, y2 - y1)

                # Convert hand range to system volume range
                vol = np.interp(length, [hmin, hmax], [minVol, maxVol])
                volume.SetMasterVolumeLevel(vol, None)  # ✅ Change system volume

                # UI Updates
                volBar = np.interp(vol, [minVol, maxVol], [400, 150])
                volPer = np.interp(vol, [minVol, maxVol], [0, 100])

                cv2.rectangle(img, (30, 150), (55, 400), (209, 206, 0), 3)
                cv2.rectangle(img, (30, int(volBar)), (55, 400), (215, 255, 127), cv2.FILLED)
                cv2.putText(img, f'{int(volPer)}%', (25, 430), cv2.FONT_HERSHEY_COMPLEX, 0.9, (209, 206, 0), 3)

    # CURSOR CONTROL MODE
    if mode == 'Cursor':
        active = 1
        putText(img, mode)
        cv2.rectangle(img, (110, 20), (620, 350), (255, 255, 255), 3)

        if fingers[1:] == [0, 0, 0, 0]:  # Only thumb down exits mode
            active = 0
            mode = 'N'
        else:
            if len(lmList) != 0:
                x1, y1 = lmList[8][1], lmList[8][2]  # Index finger tip
                w, h = pyautogui.size()

                X = int(np.interp(x1, [110, 620], [0, w - 1]))
                Y = int(np.interp(y1, [20, 350], [0, h - 1]))

                cv2.circle(img, (x1, y1), 7, (255, 255, 255), cv2.FILLED)

                pyautogui.moveTo(X, Y)  # ✅ Move mouse
                if fingers[0] == 0:  # Click when thumb is down
                    pyautogui.click()

    # FPS Calculation
    cTime = time.time()
    fps = 1 / ((cTime + 0.01) - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS:{int(fps)}', (480, 50), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)
    cv2.imshow('Silent Cue - Hand LiveFeed', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
