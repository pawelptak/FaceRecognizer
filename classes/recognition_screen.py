from kivy.uix.screenmanager import Screen
from functions.configuration import *
import tkinter as tk
from tkinter import filedialog
from functions.face_detection import *
from functions.cv2_face_recognition import *
from functions.empty_dir import *
from keras.models import load_model
from functions.cnn_face_recognition_v2 import get_prediction


class RecognitionScreen(Screen):
    image_source = ''
    file_names = ''
    recognitions_path = './recognitions/'
    face_name = 'detection'
    recognition_imgs = []
    selected_index = 0
    number_correct = 0
    number_incorrect = 0
    chosen_model = 'None'

    def __init__(self, **kw):
        super().__init__(**kw)

    def detect(self, img_source, model, f_detector, s_predictor, model_info, facenet_model):
        detection_path = os.path.join(self.recognitions_path, self.face_name)
        detected = face_detect(image_path=img_source, save_path=self.recognitions_path, face_name=self.face_name,
                               draw_points=False, face_det=f_detector, shape_pred=s_predictor)
        # if self.ids.cam_box.play:
        #   print('camera enabled')
        #  file_name = './detections/selfie.png'
        # self.ids.cam_box.export_to_png(file_name)
        # self.ids.camera_switch.trigger_action(duration=0.1)  # press button to turn off the camera
        # self.ids.face_image.load_image(file_name)
        # detected = face_detect(file_name, self.ids.name_input.text)

        print('detected: ', str(detected[1]), 'saved in:', detection_path)

        detection_results = []

        for filename in os.listdir(detection_path):
            file_path = os.path.join(detection_path, filename)
            if model_info.algorithm == 4:
                detection_results.append(
                    get_prediction(file_path=file_path, required_size=(160, 160), facenet_model=facenet_model,
                                   prediction_model=model,
                                   encoder=model_info.encoder))
            else:
                detection_results.append(recognize(file_path, model, model_info.label_dictionary))
        print(detection_results)

        if os.path.isfile(detected[0]):
            cv2_image = cv2.imread(detected[0])
            faces = detected[2]
            for i in range(len(faces)):
                cv2_image = cv2.putText(cv2_image, detection_results[i][0], (faces[i].left(), faces[i].top()),
                                        cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2)
            cv2.imwrite(detected[0], cv2_image)
            self.recognition_imgs = self.get_recognition_images(self.recognitions_path)

    def get_recognition_images(self,
                               dir_path):  # returns list of images in given directory with a name starting with 'det'
        recognitions = []
        if os.path.isdir(dir_path):
            for file_name in os.listdir(dir_path):
                if file_name.startswith('det_'):
                    file_path = os.path.join(dir_path, file_name)
                    recognitions.append(file_path)
        return recognitions

    def load_image_source(self):
        img_source = load_config()
        return img_source

    def open_file_dialog(self):
        self.ids.accuracy_text.opacity = 0
        self.number_correct = 0
        self.number_incorrect = 0
        self.selected_index = 0
        detection_path = self.recognitions_path + self.face_name + '/'
        del_all_files(self.recognitions_path)  # empty detection directory
        self.results = []

        file_names = []
        if self.ids.cam_box.play:
            file_name = './detections/selfie.png'
            self.ids.cam_box.export_to_png(filename=file_name)
            self.ids.camera_switch.trigger_action(duration=0.1)  # press button to turn off the camera
            file_names.append(file_name)
        else:
            root = tk.Tk()
            root.withdraw()

            file_names = filedialog.askopenfilenames(filetypes=[("Image files", ".jpeg .jpg .png .bmp .tiff")])

            self.get_root_window().raise_window()  # set focus on window

        if len(file_names) > 0:
            prediction_model = None
            facenet_model = None

            model, model_info = self.load_model(self.chosen_model)

            if model_info.algorithm == 4:
                facenet_model = load_model('./models/facenet_keras.h5', compile=False)

            face_detector = load_face_detector()
            shape_predictor = load_shape_predictor()
            for file_name in file_names:
                self.detect(file_name, model, face_detector, shape_predictor, model_info, facenet_model)
                del_all_files(detection_path)  # empty detection directory
            self.ids.face_image.load_image(self.recognition_imgs[self.selected_index])  # show first image

    def next_img(self):
        if self.selected_index < len(self.recognition_imgs) - 1:
            self.selected_index += 1
            self.ids.face_image.source = self.recognition_imgs[self.selected_index]
        print('index:', self.selected_index)

    def previous_img(self):
        if self.selected_index > 0:
            self.selected_index -= 1
            self.ids.face_image.source = self.recognition_imgs[self.selected_index]
        print('index:', self.selected_index)

    def rate_image(self, correct: bool):
        if self.recognition_imgs:
            if correct:
                self.number_correct += 1
                print(self.number_correct, 'correct')
            else:
                self.number_incorrect += 1
                print(self.number_incorrect, 'icorrect')
            self.next_img()
            number_rated = self.number_correct + self.number_incorrect
            accuracy = self.number_correct / len(self.recognition_imgs)
            accuracy = "{:.0%}".format(accuracy)
            if number_rated == len(self.recognition_imgs):
                self.ids.accuracy_text.opacity = 1
                self.ids.accuracy_text.text = 'Correct: ' + str(self.number_correct) + '/' + str(
                    len(self.recognition_imgs)) + ', Accuracy: ' + str(accuracy)
                print(str(self.number_correct), 'out of', str(len(self.recognition_imgs)), 'Accuracy:', str(accuracy))

    def use_webcam(self):
        if not self.ids.camera_switch.on:
            self.ids.cam_box.play = True
            self.ids.face_image.reset_image()
            self.ids.cam_box.opacity = 1
        else:
            self.ids.cam_box.play = False
            self.ids.cam_box.opacity = 0

    def load_model(self, model_name):
        model = None
        model_path = os.path.join('./models', model_name, 'model')
        model_info_path = model_path + '.info'
        model_info = pickle.load(open(model_info_path, "rb"))

        if model_info.algorithm != 4:
            if os.path.isfile(model_path):
                model = load_model_file(model_path, model_info.algorithm)
        else:
            model = pickle.load(open(model_path, 'rb'))
        print('Loaded ' + model_info.get_algorithm_name() + ' model.')
        return model, model_info

    def get_values(self):
        dir_names = []
        model_dir = './models'
        for file_name in os.listdir(model_dir):
            if os.path.isdir(os.path.join(model_dir, file_name)):
                dir_names.append(file_name)
        return dir_names

    def on_spinner_select(self, name):
        self.chosen_model = name
        self.update_values()
        print('chosen', name)

    def update_values(self):
        values = self.get_values()
        self.ids.model_spinner.values = values

        if len(values) > 0:
            if self.chosen_model == 'None':
                self.ids.model_spinner.text = values[0]
                self.chosen_model = values[0]
            else:
                self.ids.model_spinner.text = self.chosen_model

            model_info_path = os.path.join('./models', self.chosen_model, 'model.info')
            model_info = pickle.load(open(model_info_path, "rb"))
            self.ids.created_info.text = '[b]Created:[/b] ' + model_info.creation_date
            self.ids.algorithm_info.text = '[b]Algorithm:[/b] ' + model_info.get_algorithm_name()
            self.ids.labels_info.text = '[b]Labels:[/b] ' + model_info.get_labels()


    def on_pre_enter(self, *args):
        self.update_values()

