from sklearn import model_selection as cval
from sklearn.base import BaseEstimator
from sklearn.metrics import precision_score, make_scorer
import os
import cv2
import numpy as np


class FaceRecognizerModel(BaseEstimator):
    def __init__(self):
        self.model = cv2.face.EigenFaceRecognizer_create(num_components=80)

    def fit(self, X, y):
        self.model.train(X,y)

    def predict(self, T):
        return [self.model.predict(T[i])[0] for i in range(0, T.shape[0])]



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
    return np.asarray(faces,dtype=np.uint8), np.asarray(labels,dtype=np.int32)


if __name__ == '__main__':
    faces, labels = prepare_training_data('../detections')

    #cv = cval.KFold(n_splits=5, shuffle=True)
    cv = cval.StratifiedKFold(n_splits=10, shuffle=True)
    estimator = FaceRecognizerModel()

    scores = cval.cross_val_score(estimator=estimator, X=faces, y=labels, cv=cv)

