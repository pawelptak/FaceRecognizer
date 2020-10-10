from kivy.uix.screenmanager import Screen
from functions.configuration import *
import tkinter as tk
from tkinter import filedialog

class RecognitionScreen(Screen):
    image_source = ''
    file_names = ''

    def __init__(self, **kw):
        super().__init__(**kw)

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


