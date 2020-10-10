from kivy.uix.screenmanager import Screen
from functions.configuration import *
import tkinter as tk
from tkinter import filedialog
from functions.face_detection import *
from functions.face_recognition import *
from functions.empty_dir import *

class RecognitionScreen(Screen):
    image_source = ''
    file_names = ''
    recognitions_path = './recognitions/'
    face_name = 'detection'
    def __init__(self, **kw):
        super().__init__(**kw)

    def detect(self):
        detected = 'Nothing'
        detection_path = self.recognitions_path+self.face_name + '/'
        if self.ids.face_image.image_loaded:
            self.ids.face_image.source = self.file_names
            self.ids.face_image.reload()
            del_all_files(detection_path) #empty detection directory
            detected = face_detect(image_path=self.ids.face_image.source, save_path=self.recognitions_path, face_name=self.face_name, draw_landmarks=False)
        elif self.ids.cam_box.play:
            print('camera enabled')
            file_name = './detections/selfie.png'
            self.ids.cam_box.export_to_png(file_name)
            self.ids.camera_switch.trigger_action(duration=0.1)  # press button to turn off the camera
            self.ids.face_image.load_image(file_name)
            detected = face_detect(file_name, self.ids.name_input.text)

        self.ids.face_image.reload()
        print('detected: ', str(detected[1]), 'saved in:', detection_path)

        detection_results = []
        model = cv2.face.LBPHFaceRecognizer_create()
        model.read('./models/model')

        for filename in os.listdir(detection_path):
             file_path = os.path.join(detection_path, filename)
             detection_results.append(recognize(file_path, model, './detections'))
        print('123123',detection_results)
        if os.path.isfile(detected[0]):
            cv2_image = cv2.imread(detected[0])
            faces = detected[2]
            for i in range(len(faces)):
                cv2_image = cv2.putText(cv2_image, detection_results[i][0], (faces[i].left(), faces[i].top()),
                                         cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 0, 0), 2)
            cv2.imwrite(detected[0], cv2_image)
            self.ids.face_image.source = detected[0]


    def load_image_source(self):
        img_source = load_config()
        return img_source

    def open_file_dialog(self):

        root = tk.Tk()
        root.withdraw()
        #file_names = filedialog.askopenfilenames(filetypes=[("Image files", ".jpeg .jpg .png .bmp .tiff")])
        file_name = filedialog.askopenfilename(filetypes=[("Image files", ".jpeg .jpg .png .bmp .tiff")])

        self.get_root_window().raise_window()  # set focus on window

        if file_name:
        #self.file_names = file_names
            self.file_names = file_name
            self.ids.face_image.load_image(self.file_names)
            self.detect()


