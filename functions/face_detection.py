import dlib
import cv2
from functions.function_time import *
import os
from datetime import datetime

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def face_detect_hog(img_path: str, face_detector):
    try:
        img = cv2.imread(img_path)
        grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # converts to greyscale
        faces = face_detector(grey_img, 1)
        return faces
    except:
        print('Image load error or no faces detected')


def is_gpu_used():
    print(dlib.DLIB_USE_CUDA)


def load_face_detector():
    print('Loading face detector...')
    return dlib.get_frontal_face_detector()


def load_shape_predictor():
    print('Loading shape predictor...')
    return dlib.shape_predictor("./models/shape_predictor_68_face_landmarks.dat")


def face_detect_haar(img_path: str, scale_factor: float, min_neighbors: int, min_size: tuple):
    print(scale_factor, min_neighbors, min_size)
    try:

        img = cv2.imread(img_path)

        # scale_percent = 50
        # width = int(img.shape[1] * scale_percent / 100)
        # height = int(img.shape[0] * scale_percent / 100)
        # dsize = (width, height)
        # img = cv2.resize(img, dsize)
        grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # converts to greyscale
        # img = cv2.equalizeHist(img)
        faces = face_cascade.detectMultiScale(
            grey_img,
            scaleFactor=scale_factor,
            # specifying how much the image size is reduced at each image scale (default 1.5). Has to be larger than 1
            minNeighbors=min_neighbors,
            # specifying how many neighbors each candidate rectangle should have to retain it (default 5)
            minSize=min_size)  # (default 10,10)

        print(faces)
        return faces
    except:
        print('Image load error or no faces detected')


def align_face(img, d, save_path, name, shape_predictor, num):
    # img = cv2.imread(img_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    shape_predictor = shape_predictor
    landmarks = shape_predictor(img, d)
    face_chip = dlib.get_face_chip(img, landmarks, size=200)
    face_chip = cv2.normalize(face_chip, None, 0, 255, cv2.NORM_MINMAX)  # image nomralization
    face_chip = cv2.cvtColor(face_chip, cv2.COLOR_BGR2GRAY)  # converting to greyscale

    now = datetime.now()
    dt_string = now.strftime("%d%m%Y_%H%M%S")
    save_path = os.path.join(save_path, name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_name = name + '_' + dt_string + '_' + str(num) + '.jpg'
    save_path = os.path.join(save_path, file_name)

    cv2.imwrite(save_path, face_chip)  # saving detected faces as images
    return landmarks


def draw_landmarks(img, landmarks):
    for i in range(0, 68):
        x = landmarks.part(i).x
        y = landmarks.part(i).y
        cv2.circle(img, (x, y), 2, (0, 0, 255), -1)


# hog or haar face detection with alignment, returns path to image with detections, number of detetions, and detection coordinates
def face_detect(image_path, save_path, face_name, draw_points: bool, face_det,
                shape_pred, num):
    img_name = os.path.basename(image_path)

    # HAAR detection
    # detection = face_detect_haar(img, 1.5, 5, (10,10))

    # HOG detection
    detection = face_detect_hog(image_path, face_det)

    num_detected = len(detection)
    if num_detected is not 0:
        cv2_image = cv2.imread(image_path)

        # org_img_path = save_path + img_name
        org_img_path = os.path.join(save_path, img_name)
        cv2.imwrite(org_img_path, cv2_image)  # saving image without detections

        # for (x, y, w, h) in detection:
        # cv2_image = cv2.rectangle(cv2_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        landmarks = []

        for f in detection:
            print('face', f)
            landmarks.append(align_face(cv2_image, f, save_path, face_name, shape_pred, num))

        if draw_points:
            for i in landmarks:
                draw_landmarks(cv2_image, i)

        for f in detection:
            cv2_image = cv2.rectangle(cv2_image, (f.left(), f.top()), (f.right(), f.bottom()), (255, 0, 0), 2)

        detected_filename = 'det_' + img_name
        save_path = os.path.join(save_path, detected_filename)
        cv2.imwrite(save_path, cv2_image)  # saving image with detections
    else:
        save_path = 'No faces detected'
    return save_path, num_detected, detection
