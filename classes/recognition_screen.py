from kivy.uix.screenmanager import Screen
from functions.configuration import *
import tkinter as tk
from tkinter import filedialog
from functions.face_detection import *

class RecognitionScreen(Screen):
    image_source = ''
    file_names = ''

    def __init__(self, **kw):
        super().__init__(**kw)


    def detect(self):
        detected = 'Nothing'
        if self.ids.face_image.image_loaded:
            for image in self.file_names:
                self.ids.face_image.source = image
                self.ids.face_image.reload()
                detected = face_detect(self.ids.face_image.source, '')
        elif self.ids.cam_box.play:
            print('camera enabled')
            file_name = './detections/selfie.png'
            self.ids.cam_box.export_to_png(file_name)
            self.ids.camera_switch.trigger_action(duration=0.1)  # press button to turn off the camera
            self.ids.face_image.load_image(file_name)
            detected = face_detect(file_name, self.ids.name_input.text)
        if os.path.isfile(detected[0]):
            self.ids.face_image.source = detected[0]
        self.ids.face_image.reload()
        self.ids.result.text = 'Faces detected: ' + str(detected[1])
        self.ids.result.opacity = 1

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


