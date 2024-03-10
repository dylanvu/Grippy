from SpeechTranscriber import SpeechTranscriber
from gesture_recognition import handDetector
import sys
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from cursor_movement import *
from multiprocessing import Process


def video_recognition(callback=lambda: None):
    # For webcam input:
    print("Initializing video")
    cap = cv2.VideoCapture(0)

    print("Shaping video")
    cap.set(3, 1920)
    cap.set(4, 1080)

    # Camera, fixing fisheye
    cameraMatrix = np.genfromtxt("./camera_matrix.txt")
    dist = np.genfromtxt("./distortion.txt")

    print("Getting adjusted camera matrix")

    ret, frame = cap.read()
    h, w = frame.shape[:2]

    newCameraMatrix, roi = cv2. getOptimalNewCameraMatrix(cameraMatrix, dist, (w, h), 1, (w, h))

    with handDetector(maxHands=1) as hands:
        while cap.isOpened():
            # timestamp = int(time.time() * 1000) # current time in miliseconds
            # timestamp = mp.Timestamp() / 1000 # TESTING
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue
                
            # flip
            # image = cv2.flip(image, 1)
            # undistort
            image = cv2.undistort(image, cameraMatrix, dist, None, newCameraMatrix)
            x, y, w, h = roi
            image = image[y:y+h, x:x+w]
            image, landmark = hands.findHands(image)

            if landmark:
                landmark_x = landmark.x 
                landmark_y = landmark.y 
                
                # get screen size
                screen_x, screen_y = getScreenSize()
                
                # normalizing screen size
                new_screen_x = landmark_x * screen_x
                new_screen_y = landmark_y * screen_y
                
                moveCursor(new_screen_x, new_screen_y)
            
            
            # lmlist = hands.findPosition(image)
            # hands.findPosition(image)
            # print(lmlist)

            callback() # TESTINGGGGGGGGGGGGGGGGGG

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            # top_gesture, hand_landmarks = hands.recognizeGesture(mp_image, timestamp)

            cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()
    cv2.destroyAllWindows() 


def transcribe_speech():
    speech_transcriber = SpeechTranscriber(api_on=False)
    speech_transcriber.add_commands({
        ("click", "clique", "clicks", "cloak", "klick"): lambda _: clickCursor(True, 'left'),
        ("left click", "lift klick", "left-click", "left clack", "left cluck"): lambda _: clickCursor(True, 'left'),
        ("right click", "right-click", "right klick", "right clack", "rite click"): lambda _: clickCursor(True, 'right'),
        ("refresh", "refresher page",): lambda _: print("refresh page command received") ,
        ("screenshot", "screen shot", "screen pot", "screen spot", "screen jot"): lambda _: screenShot(),
        ("keyboard", "chordboard", "keyward", "keybored"): lambda phrase: keyboard(phrase),
        ("quit", "kit", "quite", "quid", "quilt", "quick"): lambda _: sys.exit()
    })
    speech_transcriber.main_loop()


def main():
    # multiprocessing: 
    # https://docs.python.org/3/library/multiprocessing.html
    # https://yakhyo.medium.com/multiprocessing-in-python-reading-from-webcam-video-using-multiprocessing-d78c15d1a7e6
    
    # Set up the webcam process
    webcam_process = Process(target=video_recognition)
    # Set up the speech transcriber process
    speech_process = Process(target=transcribe_speech)

    # Start the processes
    webcam_process.start()
    speech_process.start()

    while True:
        # If either process is no longer alive, terminate the other one
        if not webcam_process.is_alive():
            speech_process.terminate()
            break
        elif not speech_process.is_alive():
            webcam_process.terminate()
            break

    # Sleep for a bit to avoid busy waiting
    time.sleep(0.1)

    # Wait for the processes to complete (if needed)
    webcam_process.join()
    speech_process.join()

if __name__ == '__main__':
    main()