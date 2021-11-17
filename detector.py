from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import tensorflow as tf
import mediapipe as mp
import numpy as np
import time
import cv2


model = load_model(r"D:\ML\Object Detection\signLanguageClassifier_2.h5")
map = {0:"A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I", 9: "K", 10: "L", 11: "M", 12: "N", 
       13: "O", 14: "P", 15: "Q", 16: "R", 17: "S", 18: "T", 19: "U", 20: "V", 21: "W", 22: "X", 23: "Y"}

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            h, w, c = img.shape
            x_min, y_min, x_max, y_max = h, h, 0, 0

            for id, lm in enumerate(handLms.landmark):
                x, y = int(lm.x *w), int(lm.y*h)
                
                if x < x_min:
                    x_min = x

                if x > x_max:
                    x_max = x

                if y < y_min:
                    y_min = y

                if y > y_max:
                    y_max = y
            
            x_min, y_min, x_max, y_max = x_min-40, y_min-10, x_max+40, y_max+10

            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 255))

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            
            img1 = img[y_min:y_max+1, x_min:x_max+1]
            img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            img1 = cv2.resize(img1, (28, 28), interpolation=cv2.INTER_LINEAR)

            img1 = np.expand_dims(img1, axis=0)
            img1 = np.expand_dims(img1, axis=3)

            preds = model.predict(img1.astype(np.uint8)/255)
            pred = np.argmax(preds[0])

            print(map[pred])
            cv2.putText(img, str(map[pred]), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
        

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.imshow("Image", img)
    cv2.waitKey(1)
