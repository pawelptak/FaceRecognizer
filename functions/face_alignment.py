import cv2
import dlib
import os
from imutils import face_utils
from datetime import datetime

def align_face(img, d, name):
    #img = cv2.imread(img_path)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    landmarks = shape_predictor(img, d)
    (x, y, w, h) = face_utils.rect_to_bb(d)
    face_chip = dlib.get_face_chip(img, landmarks, size=200)
    face_chip = cv2.cvtColor(face_chip, cv2.COLOR_BGR2GRAY)

    now = datetime.now()
    dt_string = now.strftime("%d%m%Y_%H%M%S")
    save_path = './detections/' + name + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path += name + '_' + dt_string+'.jpg'

    cv2.imwrite(save_path, face_chip)  # saving detected faces as images



    for i in range(0, 68):
        x = landmarks.part(i).x
        y = landmarks.part(i).y
        cv2.circle(img, (x, y), 2, (0, 0, 255), -1)

