from kivy.uix.screenmanager import Screen
from functions.haar_cascade import *
import tkinter as tk
from tkinter import filedialog
from functions.empty_dir import *


class DetectionScreen(Screen):

    def __init__(self, **kw):
        super().__init__(**kw)

    def detect(self):
        del_all_files('./detections')
        if self.ids.haar_checkbox.active:
            self.haar_detect()

    def haar_detect(self):
        img = self.ids.face_image.source
        detection = face_detect_haar(img)

        self.ids.result.opacity = 1
        if detection is not None:
            self.ids.face_image.source = detection[0]
            self.ids.result.opacity = 1
            self.ids.result.text = str(detection[1]) + ' faces detected'
        else:
            self.ids.result.text = 'No faces detected'

    def open_file_dialog(self):
        root = tk.Tk()
        root.withdraw()
        file_name = filedialog.askopenfilename(filetypes=[("Image files", ".jpeg .jpg .png .bmp .tiff")])

        self.get_root_window().raise_window()  # set focus on window
        if file_name != '':
            self.ids.face_image.source = file_name
