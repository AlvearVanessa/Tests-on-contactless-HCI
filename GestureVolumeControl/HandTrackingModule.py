import cv2
import mediapipe as mp
import time

# Creating a  class
class handDetector():
    # Initialization part
    def __init__(self, mode=False, maxHands=3, detectionCon=0.5,  modelComplexity=1, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity  # Adding this
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex,
                                        self.detectionCon, self.trackCon) # Addition , self.modelComplex
        self.mpDraw = mp.solutions.drawing_utils
    #Detection part
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        # first check again
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo] #Spcify a particular hand handNo
            for id, lm in enumerate(myHand.landmark):
                # print(id,lm)
                h, w, c = img.shape # w:width, h:height, c:channel
                cx, cy = int(lm.x*w), int(lm.y*h)
                # print(id, cx, cy)
                lmList.append([id, cx, cy])
                if draw:
                    # radio 25, color purple (255, 0, 255), to paint all circle cv2.FILLED
                    cv2.circle(img, (cx, cy), 7, (0, 255, 0), cv2.FILLED) # This isfor blue (255, 0, 0)
        return lmList



def main():
    pTime = 0  # Previous time
    cTime = 0  # current time
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()
        #Send this image
        img = detector.findHands(img)
        lmList = detector.findPosition(img)

        if len(lmList) != 0:
            print(lmList[4])

        cTime = time.time()
        fps = 1/(cTime - pTime) #  to find the frames per second
        pTime = cTime

        # The position (10,70) and our font, then the color, the number 3 I dont know - something bout the skill
        # putting some color - purple (255,0,255), then the thickness 3
        # Approximated 13 fps
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3) # We wan to put the img and convert time into a string str(fps) and also round it, to not put decimal values in fps

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__": #This means if we are running this script, then do this
    main()