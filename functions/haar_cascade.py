import cv2
import os
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def face_detect_haar(img_path: str):
    try:
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)
        faces = face_cascade.detectMultiScale(
            img,
            scaleFactor=1.5,  # specifying how much the image size is reduced at each image scale
            minNeighbors=5,  # specifying how many neighbors each candidate rectangle should have to retain it
            minSize=(30, 30))

        if len(faces):
            for (x, y, w, h) in faces:
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            save_path = './detections/'+img_name
            cv2.imwrite(save_path, img)
        else:
            print('No faces detected')

        return save_path
    except:
        print('Image load error or no faces detected')