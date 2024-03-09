import cv2
import numpy as np
# function to take photos of my hand

def takePhoto():
    print("Initializing video")
    cap = cv2.VideoCapture(0)
        
    print("Shaping video")
    # Shapes video for fisheye fix adjustment
    codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    cap.set(6, codec)
    cap.set(5, 30)
    cap.set(3, 1920)
    cap.set(4, 1080)
    
    # Camera, fixing fisheye
    cameraMatrix = np.genfromtxt("./camera_matrix.txt")
    dist = np.genfromtxt("./distortion.txt")

    print("Getting adjusted camera matrix")
    
    ret, frame = cap.read()
    h, w = frame.shape[:2]
    
    newCameraMatrix, roi = cv2. getOptimalNewCameraMatrix(cameraMatrix, dist, (w, h), 1, (w, h))

    picNumber = 113

    print("Video started")

    # now, show the image
    while True:
        ret, frame = cap.read()

        # undistorts the fisheye frame
        frame = cv2.undistort(frame, cameraMatrix, dist, None, newCameraMatrix)
        x, y, w, h = roi
        frame = frame[y:y+h, x:x+w]

        cv2.imshow('frame', frame)

        key = cv2.waitKey(1)

        # press q to quit
        if key & 0xFF == ord('q'):
            break
        # press c to take a photo
        elif key & 0xFF == ord('c'):
            print(f"Taking photo {str(picNumber)}")
            # save the image
            # imageName = f"./gestures/none/{str(picNumber)}.jpg"
            imageName = f"./gestures/click/{str(picNumber)}.jpg"
            cv2.imwrite(imageName, frame)
            picNumber += 1
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    takePhoto()