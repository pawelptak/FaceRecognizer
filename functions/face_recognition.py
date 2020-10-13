import os
import pickle

import cv2
import numpy as np
from classes.face_image import FaceImage

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
        print('ddd', path)
        if os.path.isdir(path):
            images = os.listdir(path)
            for image in images:
                faces.append(cv2.imread(os.path.join(path, image), cv2.IMREAD_GRAYSCALE))
                labels.append(label)
            label += 1
    return faces, labels


def train(faces, labels):
    print("Faces: ", len(faces))
    print("Labels: ", len(labels))

    #LBPH
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()


    #EigenFaces
    #face_recognizer = cv2.face.EigenFaceRecognizer_create()

    #FisherFaces
    #face_recognizer = cv2.face.createFisherFaceRecognizer()

    face_recognizer.train(faces, np.array(labels)) #cv2 face recognizer expects numpy array

    return face_recognizer

def recognize(img, recognizer, label_dictionary):
    #put face detection function
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    label = recognizer.predict(img) #returns label and confidence (distance) - the longer the distance the less accuracy

    print(label_dictionary[label[0]], label[1])
    return label_dictionary[label[0]], label[1]


if __name__ == '__main__':
    faces, labels = prepare_training_data('../detections')
    dictionary = load_label_dictionary('../models')
    recognizer = train(faces, labels)
    recognize('../Images/add.png', recognizer, dictionary)
