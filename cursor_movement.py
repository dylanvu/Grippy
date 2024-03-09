import pyautogui
import math
import time

pyautogui.FAILSAFE = False

def getScreenSize():
    return pyautogui.size()

# function to move the cursor
def moveCursor(x_coord, y_coord):
    pyautogui.moveTo(x_coord, y_coord, duration=0.2)

# function to click
# TODO: fix problem specified below:
    # how do we know to click first, or click after movement?
    # -> async problem?
def clickCursor(click, button):
    if click:
        if button == 'left':
            pyautogui.mouseDown(button='left')
            time.sleep(0.5)
            pyautogui.mouseUp()
            
        elif button == 'right':
            pyautogui.mouseDown(button='right')
            time.sleep(0.5)
            pyautogui.mouseUp()
            
def calculate_drag_duration(start_x, start_y, end_x, end_y, base_speed=0.05, exponential_factor=0.5):
    distance = math.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
    duration = base_speed * math.pow(distance, exponential_factor)
    return duration

# function to drag cursor
# TODO: fix problem
    # maybe make the math more accurate lol
def dragCursor(x_coord, y_coord, drag_duration, button_type):
    # drag the tab
    pyautogui.dragTo(x_coord, y_coord, duration=drag_duration,button=button_type)

# function to release the mouse button
def releaseMouse():
    pyautogui.mouseUp()

def debug():
    # getting screen resolution
    print(pyautogui.size())

    # main loop, while program is running   
    while True:
        # getting current mouse position for debugging purposes 
        print(pyautogui.position())
        original_x, original_y = pyautogui.position()
        
        # pyautogui.dragTo(1289, 588, duration=1)
        
        # get input from the terminal
        input_data = input("Enter in X Coordinate, Y Coordinate separated by space: ")
        
        # parse coordinates, click flag, and button type from input
        coord_x, coord_y = input_data.split()
        coord_x = int(coord_x)
        coord_y = int(coord_y)
        
        press_input = input("Do you want to click or drag? (click/drag/no): ")
        click_flag = False
        drag_flag = False
        
        if press_input.lower() == "click":
            click_flag = True
        elif press_input.lower() == "drag":
            # calculate the duration speed
            drag_duration = calculate_drag_duration(original_x, original_y, coord_x, coord_y)
            drag_flag = True
        
        # if click_flag or drag_flag is true, then specify the button type
        if click_flag or drag_flag:
            button_type = input("What mouse input are you doing: left or right? (left/right): ")
            button_type = button_type.lower()
        
        # if click is True then click
        if click_flag:
            clickCursor(click_flag, button_type)
        
        # if drag is True then drag, drag also moves at the same time
        if drag_flag:
            dragCursor(coord_x, coord_y, drag_duration, button_type)
            
        # if drag isn't true, then move the cursor
        else:
            # move Cursor
            moveCursor(coord_x, coord_y)
        
        
        # time.sleep(3)

def main():
    # # debug function call
    # debug()
    pass
    
    
    

if __name__ == "__main__":
    main()