import os
import dlib
import cv2

def face_detect_hog(img_path: str):
    try:
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)
        face_detect = dlib.get_frontal_face_detector()
        faces = face_detect(img, 1)

        save_path = None

        if len(faces):
            for d in faces:
                img = cv2.rectangle(img, (d.left(), d.top()), (d.right(), d.bottom()), (0, 0, 255), 2)
            save_path = './detections/'+img_name
            cv2.imwrite(save_path, img)

        return save_path, len(faces)
    except:
        print('Image load error or no faces detected')