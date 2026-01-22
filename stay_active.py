import pyautogui
import time

print("Keeping screen active... Press Ctrl+C to stop")

while True:
    pyautogui.moveRel(1, 0, duration=0.1)
    pyautogui.moveRel(-1, 0, duration=0.1)
    time.sleep(60)
