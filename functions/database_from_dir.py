import os
import cv2
import dlib

#creates jpg database from pgm database (face detection and alignment included)
def create_database(dir_path):
    face_detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor("../models/shape_predictor_68_face_landmarks.dat")
    for dir_name in os.listdir(dir_path):
        print(os.listdir(dir_path))
        face_dir_path = os.path.join(dir_path, dir_name)
        if os.path.isdir(face_dir_path):
            for file_name in os.listdir(face_dir_path):
                file_path = os.path.join(face_dir_path, file_name)
                if file_path.endswith('.pgm'):
                    img = cv2.imread(file_path)
                    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = face_detect(img, face_detector, shape_predictor)
                    os.remove(file_path)
                    cv2.imwrite(file_path.replace('.pgm', '.jpg'), img)


def align_face(img, d, shape_predictor):
    # img = cv2.imread(img_path)

    shape_predictor = shape_predictor
    landmarks = shape_predictor(img, d)
    face_chip = dlib.get_face_chip(img, landmarks, size=200)
    face_chip = cv2.normalize(face_chip, None, 0, 255, cv2.NORM_MINMAX)  # image nomralization
    face_chip = cv2.cvtColor(face_chip, cv2.COLOR_BGR2GRAY)  # converting to greyscale

    return face_chip


def face_detect(img, face_det, shape_pred):

    # HOG detection
    detection = face_det(img, 1)

    landmarks = []

    for f in detection:
        print('face', f)
        img = align_face(img, f, shape_pred)

    return img


if __name__ == '__main__':
    create_database('../att_faces/')