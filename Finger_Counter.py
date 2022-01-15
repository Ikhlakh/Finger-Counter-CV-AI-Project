import cv2
import time
import os
import pyttsx3
import  Hand_Tracking_module as htm


wCam , hCam = 640, 480


cap = cv2.VideoCapture(0)
cap.set(3 , wCam)
cap.set(3 , hCam)


folderpath = "fingerimages"
myList = os.listdir(folderpath)
print(myList)

overlaylist = []
for imPath in myList:
    image = cv2.imread(f'{folderpath}/{imPath}')
    # print(f'{folderpath}/{imPath}')
    overlaylist.append(image)
# print(overlaylist)
print(len(overlaylist))

pTime = 0

Detector = htm.handDetector()

tipIds = [4 ,8 , 12 ,16 , 20]


while True:
    success, img = cap.read()
    img = Detector.findHands(img)
    lmList = Detector.findPosition(img,draw=False)
    # print(lmList)

    if len(lmList) != 0:
        fingers = []

        # Thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            # print("Thumb is  Is Open ")
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 Fingers
        for id in range(1,5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                # print("Index Finger Is Open ")
                fingers.append(1)
            else:
                fingers.append(0)
        # print(fingers)


        # Print seprate figures
        totalfingers = fingers.count(1)
        print(totalfingers)

        # Change the image according to finger
        h, w, c = overlaylist[totalfingers-1].shape
        img[250:h + 250, 0:w] = overlaylist[totalfingers-1]

        # create rectangle showing Counting
        cv2.rectangle(img, (200, 225), (0, 50), (238,130,238), cv2.FILLED)
        cv2.putText(img, str(totalfingers), (45, 200), cv2.FONT_HERSHEY_SIMPLEX,
                    5, (0,255,127), 20)



    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)


    cv2.imshow("Image", img)
    cv2.waitKey(1)


