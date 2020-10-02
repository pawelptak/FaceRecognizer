from kivy.uix.screenmanager import Screen
from functions.haar_cascade import *
from functions.configuration import *
import tkinter as tk
from tkinter import filedialog

class HAARScreen(Screen):
    image_source = ''

    def __init__(self, **kw):
        super().__init__(**kw)

    def detect(self):
        if self.ids.face_image.image_loaded:
            self.haar_detect()

    def haar_detect(self):
        self.ids.face_image.source = self.ids.face_image.original_source
        self.ids.face_image.reload()
        img = self.ids.face_image.source

        min_size = self.ids.setting_3_input.text

        detection = face_detect_haar(img, float(self.ids.setting_1_input.text), int(self.ids.setting_2_input.text), tuple(map(int, min_size.split(' '))))

        #detection = face_detect_haar(img, 1.5, 5, (10,10))
        self.ids.result.opacity = 1

        if detection[1] is not 0:
            self.ids.face_image.source = detection[0]
            self.ids.face_image.reload()
            self.ids.result.text = str(detection[1]) + ' faces detected'
        else:
            self.ids.result.text = 'No faces detected'

    def open_file_dialog(self):
        root = tk.Tk()
        root.withdraw()
        file_name = filedialog.askopenfilename(filetypes=[("Image files", ".jpeg .jpg .png .bmp .tiff")])

        self.get_root_window().raise_window()  # set focus on window
        if file_name != '':
            #self.ids.face_image.source = file_name
            self.ids.face_image.load_image(file_name)

    def enable_hog(self, value):
        if value:
            self.ids.setting_box.opacity = 1
            self.ids.setting_box.disabled = False
            self.ids.setting_1_title.text = 'Scale factor:'
            self.ids.setting_1_input.text = '1.5'
            self.ids.setting_1_input.hint = 'e.g 1.5'

            self.ids.setting_2_title.text = 'Min neighbors: '
            self.ids.setting_2_input.text = '5'
            self.ids.setting_2_input.hint = 'e.g 5'

            self.ids.setting_3_title.text = 'Min size: '
            self.ids.setting_3_input.text = '10 10'
            self.ids.setting_3_input.hint = 'e.g 10, 10'
        else:
            self.ids.setting_box.opacity = 0
            self.ids.setting_box.disabled = True

    def load_image_source(self):
        img_source = load_config()
        return img_source

    def on_pre_enter(self, *args):
        src = load_config()
        if src:
            self.ids.face_image.load_image(src)
            self.ids.result.opacity = 0

