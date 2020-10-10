from kivy.uix.screenmanager import Screen
from functions.configuration import *
import tkinter as tk
from tkinter import filedialog
from kivy.uix.camera import Camera
from functions.face_detection import *

class DetectionScreen(Screen):
    image_source = ''
    file_names = ''
    detections_path = './detections/'
    def __init__(self, **kw):
        super().__init__(**kw)

    def detect(self):
        if self.ids.name_input.text != '':
            detected = 'Nothing'
            if self.ids.face_image.image_loaded:
                for image in self.file_names:
                    self.ids.face_image.source = image
                    self.ids.face_image.reload()
                    detected = face_detect(self.ids.face_image.source, self.detections_path, self.ids.name_input.text)
            elif self.ids.cam_box.play:
                print('camera enabled')
                file_name = './detections/selfie.png'
                self.ids.cam_box.export_to_png(file_name)
                self.ids.camera_switch.trigger_action(duration=0.1)  # press button to turn off the camera
                self.ids.face_image.load_image(file_name)
                detected = face_detect(file_name, self.detections_path, self.ids.name_input.text)
            if os.path.isfile(detected[0]):
                self.ids.face_image.source = detected[0]
            self.ids.face_image.reload()
            self.ids.result.text = 'Faces detected: ' + str(detected[1])
            self.ids.result.opacity = 1


    def open_file_dialog(self):

        root = tk.Tk()
        root.withdraw()
        file_names = filedialog.askopenfilenames(filetypes=[("Image files", ".jpeg .jpg .png .bmp .tiff")])

        self.get_root_window().raise_window()  # set focus on window

        if len(file_names) > 0:
            self.file_names = file_names
            self.ids.face_image.load_image(self.file_names[0])
            self.ids.num_loaded_text.opacity = 1
            self.ids.num_loaded_text.text = str(len(self.file_names)) + ' loaded'


    def load_image_source(self):
        img_source = load_config()
        return img_source



    def use_webcam(self):

        self.ids.num_loaded_text.opacity = 0
        self.ids.image_boxlayout.remove_widget(self.ids.face_image)
        cam = Camera(x=self.ids.image_button.x, y=self.ids.image_button.y, size=self.ids.image_button.size,
                     resolution=(640, 480),
                     play=False)
        if self.ids.camera_switch.text == 'OFF':
            self.ids.image_button.add_widget(cam)
            cam.play = True
            #self.ids.cam_box.play = True
            self.ids.face_image.reset_image()
            #self.ids.cam_box.opacity = 1
            self.ids.image_button.disabled = True

        else:
            self.ids.image_button.remove_widget(cam)
            self.ids.image_button.disabled = False
