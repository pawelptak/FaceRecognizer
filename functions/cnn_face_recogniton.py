from keras.layers import Input, Lambda, Dense, Flatten
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.models import Model, Sequential
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.callbacks import EarlyStopping, History, Callback
from keras.models import load_model
from PIL import Image
from functions.face_recognition import load_label_dictionary

train_path = '../detections'
test_path = '../validation_set'

# re-size all the images to this
IMAGE_SIZE = [200, 200]




def prepare_dnn_data(dir_path):
    dirs = os.listdir(dir_path)

    X, y = list(), list()
    label = 1
    for dir_name in dirs:
        path = os.path.join(dir_path, dir_name)
        if os.path.isdir(path):
            images = os.listdir(path)
            for image in images:
                img = cv2.imread(os.path.join(path, image))
                img = cv2.resize(img, (160,160))
                X.append(img)
                y.append(label)
            label += 1
    return np.asarray(X), np.asarray(y)



def get_embedding(model, face):
    # scale pixel values
    face_pixels = face.astype('float32')
    # standardization
    mean, std = face_pixels.mean(), face_pixels.std()
    face = (face-face_pixels - mean)/std
    # transfer face into one sample (3 dimension to 4 dimension)
    sample = np.expand_dims(face, axis=0)
    # make prediction to get embedding
    yhat = model.predict(sample)
    return yhat[0]


def get_class_number(dir_path): #returns number of subdirs in given directory
    i = 0
    if os.path.isdir(dir_path):
        for file_name in os.listdir(dir_path):
            i += 1
    return i



def train():
    # add preprocessing layer to the front of VGG
    vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

    # don't train existing weights
    for layer in vgg.layers:
        layer.trainable = False

    # our layers - you can add more if you want
    x = Flatten()(vgg.output)
    # x = Dense(1000, activation='relu')(x)
    prediction = Dense(get_class_number(test_path), activation='softmax')(x)

    # create a model object
    model = Model(inputs=vgg.input, outputs=prediction)

    # view the structure of the model
    model.summary()

    # tell the model what cost and optimization method to use
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )


    from keras.preprocessing.image import ImageDataGenerator
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    training_set = train_datagen.flow_from_directory(train_path,
                                                     target_size=(IMAGE_SIZE[0], IMAGE_SIZE[1]),
                                                     batch_size=32,
                                                     class_mode='categorical')

    test_set = test_datagen.flow_from_directory(test_path,
                                                target_size=(IMAGE_SIZE[0], IMAGE_SIZE[1]),
                                                batch_size=32,
                                                class_mode='categorical')

    H = History()

    # fit the model
    r = model.fit(
      training_set,
      validation_data=test_set,
      epochs=1,
      steps_per_epoch=len(training_set),
      validation_steps=len(test_set),
      callbacks = [H, EarlyStopping(monitor='val_loss', mode='auto', restore_best_weights=True)]
    )
    # loss
    #plt.plot(H.history['loss'], label='train loss')
    #plt.plot(H.history['val_loss'], label='val loss')
    #plt.legend()
    #plt.show()
    #plt.savefig('LossVal_loss')

    # accuracies
    #plt.plot(H.history['accuracy'], label='train acc')
    #plt.plot(H.history['val_accuracy'], label='val acc')
    #plt.legend()
    #plt.show()
    #plt.savefig('AccVal_acc')

    model.save('my_model.h5')

def get_results(img_path, dictionary):
    img = cv2.imread(img_path)
    cv2.imshow('',img)
    cv2.waitKey(0)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    results = loaded_model.predict(img_array)
    label = -1
    print('results',results[0])
    for i in range(len(results[0])):
        if results[0][i] > 0.5:
            label = i + 1
            return dictionary[label]
    return 'No recognition'


if __name__ == '__main__':
    loaded_model = load_model('my_model.h5')
    print(get_results('../validation_set/pawel/pawel_val_22102020_153821.jpg', load_label_dictionary('../models/')))
