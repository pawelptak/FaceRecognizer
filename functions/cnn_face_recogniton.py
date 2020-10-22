from keras.layers.core import Flatten, Dense, Dropout
from keras.models import Model, Sequential
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16
from keras.callbacks import EarlyStopping, History, Callback
from functions.cv2_face_recognition import load_label_dictionary
from keras.models import load_model
from shutil import copyfile
from functions.empty_dir import del_everything

train_path = '../detections'
test_path = '../validation_set'

# re-size all the images to this
IMAGE_SIZE = [200, 200]


def get_class_number(dir_path):  # returns number of subdirs in given directory
    i = 0
    if os.path.isdir(dir_path):
        for _ in os.listdir(dir_path):
            i += 1
    return i


def split_dataset(dir_path, valid_percentage: int, dest_path):
    dir_name = os.path.basename(dir_path)
    if os.path.isdir(dir_path):
        i = 0
        file_list = os.listdir(dir_path)
        num_files = len(file_list)
        last_test_index = int(valid_percentage * 0.01 * num_files)
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


def cnn_train(train_path, image_size: list, epochs: int, valid_percentage: int, datasets_dir_path,
          model_path):  # valid_percentage = what percentage of train dataset files is used for validation
    # clean datasets directory
    del_everything(datasets_dir_path)

    # add preprocessing layer to the front of VGG
    vgg = VGG16(input_shape=image_size + [3], weights='imagenet', include_top=False)

    # don't train existing weights
    for layer in vgg.layers:
        layer.trainable = False

    # our layers - you can add more if you want
    x = Flatten()(vgg.output)
    # x = Dense(1000, activation='relu')(x)
    prediction = Dense(get_class_number(train_path), activation='softmax')(x)

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

    for dir_name in os.listdir(train_path):
        dir_path = os.path.join(train_path, dir_name)
        if os.path.isdir(dir_path):
            split_dataset(dir_path, valid_percentage, datasets_dir_path)

    train_set = os.path.join(datasets_dir_path, 'train')
    test_set = os.path.join(datasets_dir_path, 'test')

    from keras.preprocessing.image import ImageDataGenerator
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    training_set = train_datagen.flow_from_directory(train_set,
                                                     target_size=(image_size[0], image_size[1]),
                                                     batch_size=32,
                                                     class_mode='categorical')

    test_set = test_datagen.flow_from_directory(test_set,
                                                target_size=(image_size[0], image_size[1]),
                                                batch_size=32,
                                                class_mode='categorical')

    H = History()

    # fit the model
    r = model.fit(
        training_set,
        validation_data=test_set,
        epochs=epochs,
        steps_per_epoch=len(training_set),
        validation_steps=len(test_set),
        callbacks=[H, EarlyStopping(monitor='val_loss', patience=20, mode='auto', restore_best_weights=True)]
    )
    # loss
    # plt.plot(H.history['loss'], label='train loss')
    # plt.plot(H.history['val_loss'], label='val loss')
    # plt.legend()
    # plt.show()
    # plt.savefig('LossVal_loss')

    # accuracies
    # plt.plot(H.history['accuracy'], label='train acc')
    # plt.plot(H.history['val_accuracy'], label='val acc')
    # plt.legend()
    # plt.show()
    # plt.savefig('AccVal_acc')

    model.save(model_path)


def get_results(img_path, dictionary, model):
    img = cv2.imread(img_path)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    results = model.predict(img_array)
    label = -1
    print('results', results[0])
    #print('dic', dictionary)
    return dictionary[get_maximum_index(results[0]) + 1], results[0]


def get_maximum_index(arr: list):
    index = 0
    max = 0
    for i in range(len(arr)):
        if arr[i] > max:
            max = arr[i]
            index = i
    return index


if __name__ == '__main__':
    # train(train_path=train_path, image_size=IMAGE_SIZE, epochs=5, valid_percentage=10, datasets_dir_path='../deep_learning_datasets/', model_path='../models/dnn_model.h5')
    loaded_model = load_model('../models/dnn_model.h5')
    print(get_results('../validation_set/justyna/justyna_val_22102020_153659.jpg', load_label_dictionary('../models/'),
                      loaded_model))
