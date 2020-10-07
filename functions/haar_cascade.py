import os
import cv2
from functions.function_time import *
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@timing
def face_detect_haar(img_path: str, scale_factor: float, min_neighbors: int, min_size: tuple):
    print(scale_factor, min_neighbors, min_size)
    try:
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)

        scale_percent = 50
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dsize = (width, height)
        #img = cv2.resize(img, dsize)
        grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #converts to greyscale
        #img = cv2.equalizeHist(img)
        faces = face_cascade.detectMultiScale(
            grey_img,
            scaleFactor=scale_factor,  # specifying how much the image size is reduced at each image scale (default 1.5). Has to be larger than 1
            minNeighbors=min_neighbors,  # specifying how many neighbors each candidate rectangle should have to retain it (default 5)
            minSize=min_size) #(default 10,10)

        save_path = None

        if len(faces):
            for (x, y, w, h) in faces:
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            save_path = './detections/'+img_name
            cv2.imwrite(save_path, img)

        return save_path, len(faces)
    except:
        print('Image load error or no faces detected')


