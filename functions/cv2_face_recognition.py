import os
import pickle
import cv2
import numpy as np


# function not used anymore
def create_label_dictionary(dir_path, dictionary_savepath):
    dict = {}
    dirs = os.listdir(dir_path)
    label = 1
    for dir_name in dirs:
        dict[label] = dir_name
        label += 1
    save_path = os.path.join(dictionary_savepath, 'label_dic.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(dict, f, pickle.HIGHEST_PROTOCOL)


# function not used anymore
def load_label_dictionary(dictionary_savepath):
    save_path = os.path.join(dictionary_savepath, 'label_dic.pkl')
    with open(save_path, 'rb') as f:
        return pickle.load(f)


def prepare_training_data(dir_path):
    dirs = os.listdir(dir_path)

    faces = []
    labels = []
    label = 1
    for dir_name in dirs:
        path = os.path.join(dir_path, dir_name)
        if os.path.isdir(path):
            images = os.listdir(path)
            for image in images:
                faces.append(cv2.imread(os.path.join(path, image), cv2.IMREAD_GRAYSCALE))
                labels.append(label)
            label += 1
    return faces, labels


def load_model_file(path, algorithm: int):  # 1 - LBPH, 2 - EigenFaces, 3 - FisherFaces

    if algorithm == 1:
        # LBPH
        model = cv2.face.LBPHFaceRecognizer_create()
    elif algorithm == 2:
        # EigenFaces
        model = cv2.face.EigenFaceRecognizer_create()
    else:
        # FisherFaces
        model = cv2.face.FisherFaceRecognizer_create()

    model.read(path)
    print('Model file loaded.')
    return model


def train(faces, labels, algorithm: int):  # 1 - LBPH, 2 - EigenFaces, 3 - FisherFaces
    print("Faces: ", len(faces))
    print("Labels: ", len(labels))

    if algorithm == 1:
        # LBPH
        print('LBPH chosen')
        face_recognizer = cv2.face.LBPHFaceRecognizer_create(radius=30, neighbors=8, grid_x=8, grid_y=8, threshold=5000)
    elif algorithm == 2:
        # EigenFaces
        print('Eigenfaces chosen')
        face_recognizer = cv2.face.EigenFaceRecognizer_create(num_components=80)
    else:
        # FisherFaces
        print('Fisherfaces chosen')
        face_recognizer = cv2.face.FisherFaceRecognizer_create(num_components=0)

    face_recognizer.train(faces, np.array(labels))  # cv2 face recognizer expects numpy array
    # create_label_dictionary('./detections', './models')

    return face_recognizer


def recognize(img, recognizer, label_dictionary):
    # put face detection function
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    label, confidence = recognizer.predict(
        img)  # returns label and confidence (distance) - the longer the distance the less accuracy
    return label_dictionary[label], confidence


if __name__ == '__main__':
    faces, labels = prepare_training_data('../detections')
    dictionary = load_label_dictionary('../models')
    recognizer = train(faces, labels)
    recognize('../Images/add.png', recognizer, dictionary)
