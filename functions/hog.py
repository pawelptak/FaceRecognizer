import dlib
import cv2
from functions.function_time import *

@timing
def face_detect_hog(img_path: str):
    try:
        img = cv2.imread(img_path)
        grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # converts to greyscale
        face_detect = dlib.get_frontal_face_detector()
        faces = face_detect(grey_img, 1)
        #faces_ = []
        #if len(faces):
            #for d in faces:
                #faces_.append([d.left(), d.top(), d.right()-d.left(), d.bottom()-d.top()]) #convert hog coordinates to: x, y, w, h
        return faces
    except:
        print('Image load error or no faces detected')