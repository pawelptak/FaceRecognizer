from kivy.uix.screenmanager import Screen
import os
import tkinter as tk
from tkinter import filedialog
from functions.configuration import *

class TrainingScreen(Screen):
    photos_dir = './detections/'
    dir_name = 'None'
    img_preview = ''

    def __init__(self, **kw):
        super().__init__(**kw)
        self.dir_name = self.get_values()[0]  # show first directory on the spinner
        self.img_preview = self.get_img_preview()


    def get_img_preview(self):
        name = self.dir_name[:self.dir_name.find(' (')]
        path = os.path.join(self.photos_dir, name)
        print(os.path.join(path, os.listdir(path)[0]))
        return os.path.join(path, os.listdir(path)[0])

    def open_file_dialog(self):
        root = tk.Tk()
        root.withdraw()
        dir_name = filedialog.askdirectory()

        self.get_root_window().raise_window()  # set focus on window
        if dir_name != '':
            self.photos_dir = dir_name
            self.update_values()

    def update_values(self): #update displayed values
        self.ids.current_dir_text.text = self.photos_dir
        values = self.get_values()
        self.ids.directory_spinner.values = values
        self.ids.directory_spinner.text = values[0]
        self.ids.preview.source = self.get_img_preview()
        self.ids.preview.reload()

    def get_values(self):
        values = []
        dir_names = os.listdir(self.photos_dir)
        for name in dir_names:
            dir_name = os.path.join(self.photos_dir, name)
            if os.path.isdir(dir_name):
                num_files = 0
                for file_name in os.listdir(dir_name):
                    if file_name.endswith('.jpg'):
                        num_files += 1
                values.append(name + ' (' + str(num_files) + ')')
        print(values)
        return values


    def on_spinner_select(self, name):
        self.dir_name = name
        self.update_values()
        print('chosen', name)
