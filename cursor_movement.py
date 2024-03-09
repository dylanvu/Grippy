#! python3
import pyautogui
import time

# pyautogui.dragTo(1289, 588, duration=1)

# function to move the cursor
def moveCursor(x_coord, y_coord):
    pyautogui.moveTo(x_coord, y_coord)

# function to click, if click is True
# click is given from recognizing the gesture
def clickCursor(click: bool):
    if click == True:
        # performs a left click action
        pyautogui.mouseDown()
        time.sleep(0.5)
        pyautogui.mouseUp()
        

# function to drag, if drag is True
# if True, holds the left click mouse down
# if false, lifts up the left click
def dragCursor(drag: bool):
    if drag == True:
        # get theogui.position()
        pyautogui.mouseDown()
    
    else:
        pyautogui.mouseUp()

# function to do Right Click
def rightClickCursor(click: bool):
    if click == True:
        pyautogui.mouseDown(button='right')
        time.sleep(0.5)
        pyautogui.mouseUp()
        
# #  debug   
# while True:
#     rightClickCursor(True)
#     time.sleep(10)

