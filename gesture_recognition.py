import cv2
import mediapipe as mp
import time
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from cursor_movement import *
from segment_text import *
import matplotlib.pyplot as plt
from constants import FRAME_WIDTH, FRAME_HEIGHT

# Create a gesture recognizer instance with the live stream mode:
def print_result(result: mp.tasks.vision.GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    # print('gesture recognition result: {}'.format(result))
    # if(len(result.gestures) > 0):
    #     print('click', timestamp_ms)


    top_gesture = result.gestures[0][0] if result.gestures and result.gestures[0] else None
    hand_landmarks = result.hand_landmarks 
    # if(top_gesture or hand_landmarks):
    #     print(top_gesture)
    #     print(hand_landmarks)

""" example output:
gesture recognition result: GestureRecognizerResult(
    gestures=[[Category(index=-1, score=0.6473711133003235, display_name='', category_name='none')]], 
    handedness=[[Category(index=1, score=0.9849715828895569, display_name='Left', category_name='Left')]], 
    hand_landmarks=[[NormalizedLandmark(x=0.9173455238342285, y=0.8917664885520935, z=1.1889027007327968e-07, visibility=0.0, presence=0.0), NormalizedLandmark(x=0.8654500246047974, y=0.8364646434783936, z=-0.027947193011641502, visibility=0.0, presence=0.0), NormalizedLandmark(x=0.8313690423965454, y=0.7437697649002075, z=-0.050038088113069534, visibility=0.0, presence=0.0), NormalizedLandmark(x=0.8088149428367615, y=0.6775509119033813, z=-0.07226637005805969, visibility=0.0, presence=0.0), NormalizedLandmark(x=0.7998332977294922, y=0.6235805153846741, z=-0.09507430344820023, visibility=0.0, presence=0.0), NormalizedLandmark(x=0.8747177720069885, y=0.6027312278747559, z=-0.023846453055739403, visibility=0.0, presence=0.0), NormalizedLandmark(x=0.865658700466156, y=0.49776098132133484, z=-0.052233919501304626, visibility=0.0, presence=0.0), NormalizedLandmark(x=0.8587172031402588, y=0.43357402086257935, z=-0.07583887875080109, visibility=0.0, presence=0.0), NormalizedLandmark(x=0.8531180024147034, y=0.384832501411438, z=-0.09255510568618774, visibility=0.0, presence=0.0), NormalizedLandmark(x=0.9068267345428467, y=0.5981228351593018, z=-0.03037467785179615, visibility=0.0, presence=0.0), NormalizedLandmark(x=0.8842515349388123, y=0.4750553369522095, z=-0.06482052057981491, visibility=0.0, presence=0.0), NormalizedLandmark(x=0.8599724769592285, y=0.41445350646972656, z=-0.0890500620007515, visibility=0.0, presence=0.0), NormalizedLandmark(x=0.8376553654670715, y=0.3716445565223694, z=-0.1018543541431427, visibility=0.0, presence=0.0), NormalizedLandmark(x=0.9415385723114014, y=0.612671971321106, z=-0.041673287749290466, visibility=0.0, presence=0.0), NormalizedLandmark(x=0.9312995076179504, y=0.49375075101852417, z=-0.07263893634080887, visibility=0.0, presence=0.0), NormalizedLandmark(x=0.9160252213478088, y=0.4228614866733551, z=-0.08875341713428497, visibility=0.0, presence=0.0), NormalizedLandmark(x=0.8989604711532593, y=0.37063342332839966, z=-0.09596656262874603, visibility=0.0, presence=0.0), NormalizedLandmark(x=0.9787424206733704, y=0.6434886455535889, z=-0.05546530708670616, visibility=0.0, presence=0.0), NormalizedLandmark(x=0.9857996702194214, y=0.5507894158363342, z=-0.07815904170274734, visibility=0.0, presence=0.0), NormalizedLandmark(x=0.9832339286804199, y=0.48390743136405945, z=-0.0873035416007042, visibility=0.0, presence=0.0), NormalizedLandmark(x=0.974859893321991, y=0.42474281787872314, z=-0.09188215434551239, visibility=0.0, presence=0.0)]], 
    hand_world_landmarks=[[Landmark(x=0.002034332137554884, y=0.08360659331083298, z=-0.0038515999913215637, visibility=0.0, presence=0.0), Landmark(x=-0.02206920087337494, y=0.059724751859903336, z=-0.009316004812717438, visibility=0.0, presence=0.0), Landmark(x=-0.03774220496416092, y=0.04052324220538139, z=-0.0098055899143219, visibility=0.0, presence=0.0), Landmark(x=-0.05140763521194458, y=0.021816527470946312, z=-0.014271378517150879, visibility=0.0, presence=0.0), Landmark(x=-0.05542093515396118, y=6.18944177404046e-05, z=-0.01240287721157074, visibility=0.0, presence=0.0), Landmark(x=-0.023322230204939842, y=-0.0004086859989911318, z=0.01011299341917038, visibility=0.0, presence=0.0), Landmark(x=-0.024342374876141548, y=-0.025472844019532204, z=-0.0007243901491165161, visibility=0.0, presence=0.0), Landmark(x=-0.02736847847700119, y=-0.04149956628680229, z=-0.011332504451274872, visibility=0.0, presence=0.0), Landmark(x=-0.0338439866900444, y=-0.053839389234781265, z=-0.04320019856095314, visibility=0.0, presence=0.0), Landmark(x=-0.002677882555872202, y=-0.003092429367825389, z=0.00834675133228302, visibility=0.0, presence=0.0), Landmark(x=-0.013706620782613754, y=-0.03565262258052826, z=-0.005243957042694092, visibility=0.0, presence=0.0), Landmark(x=-0.03170483186841011, y=-0.05021890997886658, z=-0.03380384296178818, visibility=0.0, presence=0.0), Landmark(x=-0.04648694768548012, y=-0.062052566558122635, z=-0.05734166502952576, visibility=0.0, presence=0.0), Landmark(x=0.014125526882708073, y=-0.0021675352472811937, z=-0.005375005304813385, visibility=0.0, presence=0.0), Landmark(x=0.007918084040284157, y=-0.02787170186638832, z=-0.016956642270088196, visibility=0.0, presence=0.0), Landmark(x=-0.0026309206150472164, y=-0.04554267227649689, z=-0.03650364279747009, visibility=0.0, presence=0.0), Landmark(x=-0.015248340554535389, y=-0.06256142258644104, z=-0.055226199328899384, visibility=0.0, presence=0.0), Landmark(x=0.025036126375198364, y=0.009923724457621574, z=-0.020022213459014893, visibility=0.0, presence=0.0), Landmark(x=0.032596249133348465, y=-0.009264006279408932, z=-0.021393928676843643, visibility=0.0, presence=0.0), Landmark(x=0.03118841163814068, y=-0.03117087483406067, z=-0.02864767611026764, visibility=0.0, presence=0.0), Landmark(x=0.02290630154311657, y=-0.049151014536619186, z=-0.04246221482753754, visibility=0.0, presence=0.0)]
])
"""

class handDetector():
    def __init__(self,mode=False,maxHands=2,modelComplexity=1,detectionCon=0.5,trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.modelComplex,self.detectionCon,self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # gesture recognizer
        # self.base_options = python.BaseOptions(model_asset_path='clicking.task')
        self.base_options = python.BaseOptions(model_asset_buffer = open('./gesture_recognizer.task', "rb").read())
        self.options = vision.GestureRecognizerOptions(
            base_options=self.base_options,
            # canned_gestures_classifier_options = ["Thumb_Down"],
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            result_callback=print_result)
        self.recognizer = vision.GestureRecognizer.create_from_options(self.options)
        # https://developers.google.com/mediapipe/api/solutions/python/mp/tasks/vision/GestureRecognizerOptions

        self.results = None

    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.mpHands.__exit__(exc_type, exc_val, exc_tb)
        self.mpDraw.__exit__(exc_type, exc_val, exc_tb)
        self.recognizer.__exit__(exc_type, exc_val, exc_tb)

    def findHands(self,img,draw=True):
        img.flags.writeable = False
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img)
        # print(self.results.multi_hand_landmarks)
        # draw hand annotations on the image
        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        index_landmark = None
        
        if self.results.multi_hand_landmarks:
            for i, handlms in enumerate(self.results.multi_hand_landmarks):   
                '''
                    '''
                if draw:
                    self.mpDraw.draw_landmarks(
                        img,
                        handlms,
                        self.mpHands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style())
                    
                    # iterate through the landmarks to get the 8th one
                    for landmark_index, landmark in enumerate(handlms.landmark):
                        if landmark_index == 8:
                            index_landmark = landmark
                            # unnormalize it
                            index_landmark.x = index_landmark.x * FRAME_WIDTH
                            index_landmark.y = index_landmark.y * FRAME_HEIGHT

        return (img, index_landmark)
    
    # def findPosition(self,img,handNo=0,draw=True):
        
    #     lmlist = []
    #     if self.results.multi_hand_landmarks:
    #         myHand=self.results.multi_hand_landmarks[handNo]

    #         for id,lm in enumerate(myHand.landmark):
    #             h,w,c = img.shape
    #             cx,cy = int(lm.x*w),int(lm.y*h)
    #             #print(id,cx,cy)
    #             lmlist.append([id,cx,cy])
    #             if draw:
    #                 cv2.circle(img,(cx,cy),5,(255,0,255),cv2.FILLED)

    #     return lmlist
    def findPosition(self,img,handNo=0,draw=True):
        print(self.results.multi_hand_landmarks)

    def recognizeGesture(self, mg_img, frame_timestamp_ms):
        # Send live image data to perform gesture recognition.
        # The results are accessible via the `result_callback` provided in
        # the `GestureRecognizerOptions` object.
        # The gesture recognizer must be created with the live stream mode.
        recognition_result = self.recognizer.recognize_async(mg_img, frame_timestamp_ms)
        # STEP 4: Recognize gestures in the input image.
        # recognition_result = self.recognizer.recognize(mg_img)
        top_gesture = recognition_result.gestures[0][0] if recognition_result and recognition_result.gestures and recognition_result.gestures[0] else None
        hand_landmarks = recognition_result.hand_landmarks if recognition_result and recognition_result.hand_landmarks else None
        if(top_gesture or hand_landmarks):
            print(top_gesture)
            print(hand_landmarks)

        # ex: recognition_result = GestureRecognizerResult(gestures=[], handedness=[], hand_landmarks=[], hand_world_landmarks=[])

        top_gesture, hand_landmarks = None, None
        return (top_gesture, hand_landmarks)
    
    


# For webcam input:
print("Initializing video")
cap = cv2.VideoCapture(0)

print("Shaping video")
cap.set(3, FRAME_WIDTH)
cap.set(4, FRAME_HEIGHT)

# Camera, fixing fisheye
cameraMatrix = np.genfromtxt("./camera_matrix.txt")
dist = np.genfromtxt("./distortion.txt")

print("Getting adjusted camera matrix")

ret, frame = cap.read()
h, w = frame.shape[:2]

newCameraMatrix, roi = cv2. getOptimalNewCameraMatrix(cameraMatrix, dist, (w, h), 1, (w, h))

M = None
warped_hull_points = []

print("Opening video feed")

with handDetector(maxHands=1) as hands:
    while cap.isOpened():
        timestamp = int(time.time() * 1000) # current time in miliseconds

        # timestamp = mp.Timestamp() / 1000 # TESTING
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue
            
        # undistort
        image = cv2.undistort(image, cameraMatrix, dist, None, newCameraMatrix)
        x, y, w, h = roi
        image = image[y:y+h, x:x+w]
        if M is None:
            print("Getting segmentation matrix")
            # get the segmentation matrix
            M, warped_hull_points = segmentImage(image)
        # get coordinates
        image, landmark = hands.findHands(image)
        # segment
        image, landmark = applySegmentation(M, image, landmark, warped_hull_points)

        # TODO: check to see if coordinate
        if landmark:
            landmark_x = landmark[0]
            landmark_y = landmark[1] 
            
            # get screen size
            screen_x, screen_y = getScreenSize()

            # normalizing screen size
            # hardcode an offset
            new_screen_x = landmark_x * screen_x - 234
            new_screen_y = landmark_y * screen_y - 259

            # print("SCREEN:", [new_screen_x, new_screen_y])

            # print("OFFSET SCREEN:", [new_screen_x - 234, new_screen_y - 259])

            moveCursor(new_screen_x, new_screen_y)
        
        
        # lmlist = hands.findPosition(image)
        # hands.findPosition(image)
        # print(lmlist)


        # mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        # top_gesture, hand_landmarks = hands.recognizeGesture(mp_image, timestamp)

        cv2.imshow('MediaPipe Hands', image)
        # plt.imshow(image)
        # plt.axis('off')  # Hide the axes
        # plt.show()
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
cv2.destroyAllWindows() 