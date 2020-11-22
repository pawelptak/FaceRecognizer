from kivy.uix.screenmanager import Screen
import tkinter as tk
from tkinter import filedialog
from functions.configuration import *
from functions.cv2_face_recognition import *
from functions.empty_dir import *
from functions.cnn_face_recognition_v2 import train_model, cnn_cross_validation_train
from classes.model import Model
from concurrent.futures import ThreadPoolExecutor

class TrainingScreen(Screen):
    photos_dir = './detections/'
    deep_learning_model_path = './models/dnn_model.h5'
    model_files_path = './models/'
    dir_name = 'None'  # chosen directory
    num_files = '0'
    img_preview = ''

    def __init__(self, **kw):
        super().__init__(**kw)

    def get_img_preview(self):
        name = self.dir_name
        path = os.path.join(self.photos_dir, name)
        first_img_path = os.path.join(path, os.listdir(path)[0])
        if os.path.isfile(first_img_path):
            return first_img_path
        else:
            return './Images/icon.png'

    def open_file_dialog(self):
        root = tk.Tk()
        root.withdraw()
        dir_name = filedialog.askdirectory()

        self.get_root_window().raise_window()  # set focus on window
        if dir_name != '':
            self.photos_dir = dir_name
            files = os.listdir(dir_name)
            chosen = self.get_first_image_dir(files)
            if chosen is not None:
                self.dir_name = os.path.basename(chosen)
            else:
                self.dir_name = 'None'
            self.update_values()

    def get_first_image_dir(self, dir_list):  # takes a directory list returns first directory with images
        for dir in dir_list:
            if self.get_file_number(dir) > 0:
                return dir

    def update_values(self):  # update displayed values
        self.img_preview = self.get_img_preview()
        self.num_files = self.get_file_number(self.dir_name)
        self.ids.preview.source = self.img_preview
        self.ids.preview.reload()

        self.ids.current_dir_text.text = self.photos_dir
        values = self.get_values()
        self.ids.directory_spinner.values = values
        self.ids.directory_spinner.text = self.dir_name
        self.ids.num_files.text = str(self.num_files) + ' files'



    def get_dir_names(self):
        names = []
        dir_names = os.listdir(self.photos_dir)
        for name in dir_names:
            dir_name = os.path.join(self.photos_dir, name)
            if os.path.isdir(dir_name):
                if self.get_file_number(name) > 0:
                    names.append(name)
        return names

    def get_values(self):
        dir_names = self.get_dir_names()
        return dir_names

    def get_file_number(self, dir_name):  # returns number of jpg files in directory
        n = 0
        path = os.path.join(self.photos_dir, dir_name)
        if os.path.isdir(path):
            for file_name in os.listdir(path):
                if file_name.endswith('.jpg') or file_name.endswith('.png'):
                    n += 1
        return n

    def on_spinner_select(self, name):
        self.dir_name = name
        self.num_files = self.get_file_number(name)
        self.update_values()
        print('chosen', name)

    def get_checkbox_value(self):
        if self.ids.lbph_checkbox.active:
            return 1
        elif self.ids.eigen_checkbox.active:
            return 2
        elif self.ids.fisher_checkbox.active:
            return 3
        elif self.ids.cnn_checkbox.active:
            return 4
        else:
            return 0

    def load_checkbox_value(self):
        algorithm = load_config()
        if algorithm == 1:
            self.ids.lbph_checkbox.active = True
        elif algorithm == 2:
            self.ids.eigen_checkbox.active = True
        elif algorithm == 3:
            self.ids.fisher_checkbox.active = True
        elif algorithm == 4:
            self.ids.cnn_checkbox.active = True

    def begin_training(self):
        algorithm = self.get_checkbox_value()
        if algorithm != 0:
            accuracy = None
            splits = self.ids.cv_checkbox.text
            cv_result = None
            saved_model_name = ''
            if algorithm == 4:
                if self.ids.cv_checkbox.text:
                    cv_result = cnn_cross_validation_train(images_source_path=self.photos_dir, facenet_model_path=os.path.join(self.model_files_path, 'facenet_keras.h5'), num_splits=int(splits))
                else:
                    model, encoder, accuracy = train_model(images_source_path=self.photos_dir,
                                facenet_model_path=os.path.join(self.model_files_path, 'facenet_keras.h5'),
                                valid_percentage=10)
                    dnn_model = Model(name=self.ids.name_input.text, algorithm=algorithm, encoder=encoder, train_set_dir=self.photos_dir,
                                      save_dir=self.model_files_path)
                    pickle.dump(model, open(os.path.join(dnn_model.save_path, 'model'), 'wb'))
                    saved_model_name = dnn_model.name
            else:
                if self.ids.cv_checkbox.text:
                    cv_result = cv2_cross_validation_train(images_source_path=self.photos_dir, algorithm=algorithm, num_splits=int(splits))
                else:
                    model, accuracy = train(images_source_path=self.photos_dir, algorithm=algorithm)
                    cv2_model = Model(name=self.ids.name_input.text, algorithm=algorithm, encoder=None, train_set_dir=self.photos_dir, save_dir=self.model_files_path)
                    model.write(os.path.join(cv2_model.save_path, 'model'))
                    saved_model_name = cv2_model.name
            if accuracy is not None:
                str_accuracy = "{0:.0%}".format(accuracy)
                self.ids.result_text.text = 'Saved as: ' + saved_model_name + '. Validation accuracy: ' + str_accuracy
            else:
                self.ids.result_text.text = str(splits) + '-fold cross validation accuracy: ' + cv_result
            self.ids.result_text.opacity = 1
            self.ids.train_button.text = 'Begin training'
            self.ids.train_button.disabled = False
            save_settings(algorithm)

    def start_training_thread(self):
        self.ids.result_text.opacity = 0
        self.ids.train_button.text = 'Training'
        self.ids.train_button.disabled = True
        executor = ThreadPoolExecutor(max_workers=1)
        executor.submit(self.begin_training)

    def on_pre_enter(self, *args):
        self.load_checkbox_value()
        del_all_files('./detections')
        directories = self.get_values()
        if directories:
            first_dir = directories[0]
            if first_dir is not None:
                self.dir_name = first_dir  # show first directory on the spinner
        self.update_values()
