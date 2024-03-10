# PyPointer

## Inspiration

PyPointer draws inspiration from a video showcasing a gaming setup devoid of a keyboard and mouse. Inspired by the lack of use of traditional tools, PyPointer challenges the concept of conventional computer interaction as all you need is your hand and voice.

This idea was chosen due to our team's drive to innovate. We aim to harness both new and traditional technologies and strategies at every Hackathon, pushing boundaries to create projects that defy the constraints of a 24-36 hour time limit.

## Features

- Control your cursor with just your fingertips
- Execute commands via speech-to-text
- Activate certain commands by saying key words (e.g., saying "click" performs a mouse left-click action)

## How We Built It

- **Image Segmentation**: To segment the computer display, we used a modification of Meta's SAM (Segment Anything Model) and text encoders. Initially, we used PCA analysis and eigenvectors to calculate bounding boxes but realized that it did not account for perspective shift. We solved this issue using OpenCV contour detection and the convex hull algorithm.

- **Finger Recognition**: To recognize the index finger being used as a cursor, we utilized OpenCV with MediaPipe to detect if a hand is in view.

- **Voice Commands**: Voice commands are recognized by OpenAI's Whisper, a machine learning model for speech recognition.

- **Cursor and Keyboard Control**: The control of the cursor and keyboard is achieved using the PyAutoGUI library.

## Challenges We Faced

- **Multithreading in Python**: Python is single-threaded, making it difficult to utilize two processes simultaneously compared to other languages. This resulted in issues running the OpenCV webcam with MediaPipe and our speech-to-text simultaneously.

- **Perspective Distortion**: To recognize the location of a computer screen, we used image segmentation. However, the images contained perspective distortion, causing inconsistencies in the coordinate system used to match the cursor movement with the finger.

- **Pivot from Gestures to Voice Commands**: Our original idea was to use gestures as shortcuts for controls. However, we realized it was challenging to perform both a gesture and position the cursor simultaneously. Due to time constraints and our current knowledge, we decided to pivot to voice commands for easier accessibility.

## Accomplishments

- Accurate laptop screen identification in image segmentation
- Rough reading of finger movement translated into cursor movement
- Solving the multithreading issue in Python by utilizing a multiprocessing library

## Lessons Learned

- Image segmentation techniques
- Handling multiple processes concurrently in Python using the multiprocessing library
- Utilization of new libraries, such as PyAutoGUI, and understanding their functionality

## Future Enhancements

- Implementing gesture recognition for more intuitive controls
- Improving the accuracy of coordinate reading for smoother finger/cursor movement
- Adjusting the PyAutoGUI drag speed equation for a more seamless interaction when dragging tabs/programs on the screen