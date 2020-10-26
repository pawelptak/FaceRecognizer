from kivy.uix.screenmanager import Screen
from functions.configuration import *
import tkinter as tk
from tkinter import filedialog
from functions.face_detection import *
from functions.cv2_face_recognition import *
from functions.cnn_face_recogniton_v1 import get_results
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
    algorithm = 0

    def __init__(self, **kw):
        super().__init__(**kw)

    def detect(self, img_source, model, f_detector, s_predictor, p_model, label_dic):
        detection_path = os.path.join(self.recognitions_path, self.face_name)
        detected = face_detect(image_path=img_source, save_path=self.recognitions_path, face_name=self.face_name, draw_points=False, face_det=f_detector, shape_pred=s_predictor)
        #if self.ids.cam_box.play:
         #   print('camera enabled')
          #  file_name = './detections/selfie.png'
           # self.ids.cam_box.export_to_png(file_name)
            #self.ids.camera_switch.trigger_action(duration=0.1)  # press button to turn off the camera
            #self.ids.face_image.load_image(file_name)
            #detected = face_detect(file_name, self.ids.name_input.text)

        self.ids.face_image.reload()
        print('detected: ', str(detected[1]), 'saved in:', detection_path)

        detection_results = []

        for filename in os.listdir(detection_path):
            file_path = os.path.join(detection_path, filename)
            if self.algorithm == 4:
               #detection_results.append(get_results(file_path, label_dictionary, model))
               detection_results.append(get_prediction(file_path=file_path, required_size=(160, 160), model=model, prediction_model=p_model, encoder_path='./models/'))
            else:
                detection_results.append(recognize(file_path, model, label_dic))
        print(detection_results)

        if os.path.isfile(detected[0]):
            cv2_image = cv2.imread(detected[0])
            faces = detected[2]
            for i in range(len(faces)):
                cv2_image = cv2.putText(cv2_image, detection_results[i][0], (faces[i].left(), faces[i].top()),
                                         cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2)
            cv2.imwrite(detected[0], cv2_image)
            self.recognition_imgs = self.get_recognition_images(self.recognitions_path)




    def get_recognition_images(self, dir_path): #returns list of images in given directory with a name starting with 'det'
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
            label_dictionary = load_label_dictionary('./models/')
            if self.algorithm == 4:
                #model = load_model('./models/dnn_model.h5')
                model = load_model('./models/facenet_keras.h5', compile=False)
                prediction_model = pickle.load(open('./models/dnn_modelv2.h5', 'rb'))
            else:
                model = load_model_file('./models/cv2_model', algorithm=self.algorithm)
            face_detector = load_face_detector()
            shape_predictor = load_shape_predictor()
            for file_name in file_names:
                self.detect(file_name, model, face_detector, shape_predictor, prediction_model, label_dictionary)
                del_all_files(detection_path)  # empty detection directory
            self.ids.face_image.load_image(self.recognition_imgs[self.selected_index]) #show first image

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
                print(self.number_incorrect,'icorrect')
            self.next_img()
            number_rated = self.number_correct + self.number_incorrect
            accuracy = self.number_correct/len(self.recognition_imgs)
            accuracy = "{:.0%}".format(accuracy)
            if number_rated == len(self.recognition_imgs):
                self.ids.accuracy_text.opacity = 1
                self.ids.accuracy_text.text = 'Correct: ' + str(self.number_correct) + '/' + str(len(self.recognition_imgs)) + ', Accuracy: ' + str(accuracy)
                print(str(self.number_correct),'out of', str(len(self.recognition_imgs)),'Accuracy:',str(accuracy))

    def on_pre_enter(self, *args):
        algorithm = load_config()
        self.algorithm = algorithm
        self.ids.algorithm_text.text = 'Algorithm: '
        if algorithm == 1:
            self.ids.algorithm_text.text += 'LBPH'
        elif algorithm == 2:
            self.ids.algorithm_text.text += 'Eigenfaces'
        elif algorithm == 3:
            self.ids.algorithm_text.text += 'Fisherfaces'
        elif algorithm == 4:
            self.ids.algorithm_text.text += 'DNN'

    def use_webcam(self):
        if not self.ids.camera_switch.on:
            self.ids.cam_box.play = True
            self.ids.face_image.reset_image()
            self.ids.cam_box.opacity = 1
        else:
            self.ids.cam_box.play = False
            self.ids.cam_box.opacity = 0


