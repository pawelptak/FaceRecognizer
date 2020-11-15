import os
import cv2
from numpy import asarray, expand_dims
from shutil import copyfile
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


# load images and extract faces for all images in a directory
def load_faces(directory, required_size=(160, 160)):
    faces = list()
    # enumerate files
    for filename in os.listdir(directory):
        # path
        path = os.path.join(directory, filename)
        # get face
        face = cv2.imread(path)
        face = cv2.resize(face, required_size)
        # store
        faces.append(face)
    return faces


# load a dataset that contains one subdir for each class that in turn contains images
def load_dataset(directory):
    X, y = list(), list()
    # enumerate folders, on per class
    for subdir in os.listdir(directory):
        # path
        path = os.path.join(directory, subdir)
        # skip any files that might be in the dir
        if not os.path.isdir(path):
            continue
        # load all faces in the subdirectory
        faces = load_faces(path)
        # create labels
        labels = [subdir for _ in range(len(faces))]
        # summarize progress
        print('>loaded %d examples for class: %s' % (len(faces), subdir))
        # store
        X.extend(faces)
        y.extend(labels)
    return asarray(X), asarray(y)


# not used, just for testing
def split_dataset(dir_path, valid_percentage: int, dest_path):
    dir_name = os.path.basename(dir_path)

    if os.path.isdir(dir_path):
        if not os.path.exists(os.path.join(dest_path, 'test')):
            os.makedirs(os.path.join(dest_path, 'test'))

        if not os.path.exists(os.path.join(dest_path, 'train')):
            os.makedirs(os.path.join(dest_path, 'train'))

        i = 0
        file_list = os.listdir(dir_path)
        num_files = len(file_list)
        last_test_index = int(valid_percentage * 0.01 * num_files)
        if last_test_index == 0:
            print('Error. More training images needed.')
        print('From', num_files, 'in', dir_name, last_test_index, '(' + str(valid_percentage) + '%)', 'for validation')
        for i in range(num_files):
            file_path = os.path.join(dir_path, file_list[i])
            if i < last_test_index:
                destination_dir = os.path.join(dest_path, 'test', dir_name)
                if not os.path.exists(destination_dir):
                    os.makedirs(destination_dir)
                destination_path = os.path.join(destination_dir, file_list[i])
                copyfile(file_path, destination_path)
            else:
                destination_dir = os.path.join(dest_path, 'train', dir_name)
                if not os.path.exists(destination_dir):
                    os.makedirs(destination_dir)
                destination_path = os.path.join(destination_dir, file_list[i])
                copyfile(file_path, destination_path)


# get the face embedding for one face
def get_embedding(model, face_pixels):
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    yhat = model.predict(samples)
    return yhat[0]


def to_embedding(model, dataset):  # convert each face in the dataset to an embedding
    new_x = list()
    for face_pixels in dataset:
        embedding = get_embedding(model, face_pixels)
        new_x.append(embedding)
    return asarray(new_x)


def train_model(images_source_path, facenet_model_path, valid_percentage=10):
    # load train dataset
    X, y = load_dataset(images_source_path)
    trainX, testX, trainy, testy = train_test_split(X, y, test_size=float(valid_percentage/100))
    print(trainX.shape, trainy.shape)
    print(testX.shape, testy.shape)

    # load the facenet model
    model = load_model(facenet_model_path, compile=False)
    print('Loaded Model')
    print('Creating embeddings...')
    trainX = to_embedding(model, trainX)
    testX = to_embedding(model, testX)
    print(testX.shape)

    # normalize input vectors
    in_encoder = Normalizer(norm='l2')
    trainX = in_encoder.transform(trainX)
    testX = in_encoder.transform(testX)

    # label encode targets
    out_encoder = LabelEncoder()
    out_encoder.fit(trainy)

    trainy = out_encoder.transform(trainy)
    testy = out_encoder.transform(testy)

    # fit model
    prediction_model = SVC(kernel='linear', probability=True)
    print('Training model...')
    prediction_model.fit(trainX, trainy)

    # predict
    yhat_train = prediction_model.predict(trainX)
    yhat_test = prediction_model.predict(testX)
    # score
    score_train = accuracy_score(trainy, yhat_train)
    score_test = accuracy_score(testy, yhat_test)
    # summarize
    print('Accuracy: train=%.3f, test=%.3f' % (score_train * 100, score_test * 100))

    return prediction_model, out_encoder, score_test


def cnn_cross_validation_train(images_source_path, facenet_model_path, num_splits):
    X, y = load_dataset(images_source_path)

    accuracy_sum = 0


    kf = KFold(n_splits=num_splits)
    model = load_model(facenet_model_path, compile=False)

    for train_index, test_index in kf.split(X):
        trainX, testX = X[train_index], X[test_index]
        trainy, testy = y[train_index], y[test_index]
        print(testy)
        print(trainy)
        trainX = to_embedding(model, trainX)
        testX = to_embedding(model, testX)

        # normalize input vectors
        in_encoder = Normalizer(norm='l2')
        trainX = in_encoder.transform(trainX)
        testX = in_encoder.transform(testX)

        out_encoder = LabelEncoder()
        out_encoder.fit(trainy)

        trainy = out_encoder.transform(trainy)
        testy = out_encoder.transform(testy)

        prediction_model = SVC(kernel='linear', probability=True)
        prediction_model.fit(trainX, trainy)

        yhat_test = prediction_model.predict(testX)

        score_test = accuracy_score(testy, yhat_test)
        print('cv accuracy:', score_test)
        accuracy_sum += score_test
    mean = float(accuracy_sum/num_splits)
    print(str(num_splits) + '-fold cross validation accuracy:', str(mean))
    return mean



def get_prediction(file_path, required_size, facenet_model, prediction_model, encoder):
    face_pixels = cv2.imread(file_path)
    face_pixels = cv2.resize(face_pixels, required_size)
    face_pixels = face_pixels.astype('float32')

    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std

    samples = expand_dims(face_pixels, axis=0)
    # Face embeddings collected
    yhat = facenet_model.predict(samples)

    # comparing the embeddings
    yhat_class = prediction_model.predict(yhat)

    # Retrieving the probability of the prediction
    yhat_prob = prediction_model.predict_proba(yhat)
    class_index = yhat_class[0]
    class_probability = yhat_prob[0, class_index] * 100
    out_encoder = encoder
    predict_names = out_encoder.inverse_transform(yhat_class)
    threshold = 99.00
    if class_probability < threshold:
        print('Predicted: %s (%.3f)' % ('unknown', class_probability))
        return 'unknown', class_probability
    else:
        print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
        return predict_names[0], class_probability

# train_model(images_source_path='../detections/',
#            images_destination_path='../deep_learning_datasets',
#            facenet_model_path='../facenet_keras.h5',
#            model_save_dir='../models/',
#            valid_percentage=10)

# img_dir = '../validation_set/'
# model = load_model('../facenet_keras.h5', compile=False)
# Loading FaceEmbedding model file
# prediction_model = pickle.load(open('../models/dnn_modelv2.h5', 'rb'))


# get_prediction('../validation_set/testy_25102020_203721.jpg', (160,160), model, prediction_model, encoder_path='../models/')
