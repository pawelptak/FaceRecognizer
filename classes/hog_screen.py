from kivy.uix.screenmanager import Screen
from functions.hog import *
import tkinter as tk
from tkinter import filedialog
from functions.configuration import *

class HOGScreen(Screen):
    def __init__(self, **kw):
        super().__init__(**kw)

    def detect(self):
        if self.ids.face_image.image_loaded:
            self.hog_detect()

    def hog_detect(self):
        self.ids.face_image.source = self.ids.face_image.original_source
        self.ids.face_image.reload()
        img = self.ids.face_image.source

        #min_size = self.ids.setting_3_input.text

        detection = face_detect_hog(img)

        #detection = face_detect_haar(img, 1.5, 5, (10,10))
        self.ids.result.opacity = 1

        if detection:
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

    def on_pre_enter(self, *args):
        src = load_config()
        if src:
            self.ids.face_image.load_image(src)
            self.ids.result.opacity = 0

