from kivy.uix.screenmanager import Screen
import os
import tkinter as tk
from tkinter import filedialog
from functions.configuration import *

class TrainingScreen(Screen):
    photos_dir = './detections/'
    dir_name = 'None' #chosen directory
    num_files = '0'
    img_preview = ''

    def __init__(self, **kw):
        super().__init__(**kw)
        first_dir = self.get_values()[0]
        if first_dir is not None:
            self.dir_name = first_dir  # show first directory on the spinner
        else:
            self.dir_name = 'None'
        self.num_files = self.get_file_number(self.dir_name)
        img_source = self.get_img_preview()
        self.img_preview = img_source


    def get_img_preview(self):
        name = self.dir_name
        if name != 'None':
            path = os.path.join(self.photos_dir, name)
            return os.path.join(path, os.listdir(path)[0])
        else:
            return './Images/icon.png'

    def open_file_dialog(self):
        root = tk.Tk()
        root.withdraw()
        dir_name = filedialog.askdirectory()

        self.get_root_window().raise_window()  # set focus on window
        if dir_name != '':
            self.photos_dir = dir_name
            files = os.listdir(dir_name)
            chosen = self.get_first_image_dir(files)
            if chosen is not None:
                self.dir_name = os.path.basename(chosen)
            else:
                self.dir_name = 'None'
            self.update_values()

    def get_first_image_dir(self, dir_list): #takes a directory list returns first directory with images
        for dir in dir_list:
            if self.get_file_number(dir) > 0:
                return dir

    def update_values(self): #update displayed values
        self.ids.current_dir_text.text = self.photos_dir
        values = self.get_values()
        self.ids.directory_spinner.values = values
        self.ids.directory_spinner.text = self.dir_name
        self.ids.num_files.text = str(self.num_files) + ' files'
        self.ids.preview.source = self.get_img_preview()
        self.ids.preview.reload()

    def get_dir_names(self):
        names = []
        dir_names = os.listdir(self.photos_dir)
        for name in dir_names:
            dir_name = os.path.join(self.photos_dir, name)
            if os.path.isdir(dir_name):
                if self.get_file_number(name) > 0:
                    names.append(name)
        print(names)
        return names

    def get_values(self):
        dir_name = self.get_dir_names()
        return dir_name

    def get_file_number(self, dir_name): #returns number of jpg files in directory
        n = 0
        path = os.path.join(self.photos_dir, dir_name)
        if os.path.isdir(path):
            for file_name in os.listdir(path):
                if file_name.endswith('.jpg'):
                    n += 1
        return n




    def on_spinner_select(self, name):
        self.dir_name = name
        self.num_files = self.get_file_number(name)
        self.update_values()
        print('chosen', name)