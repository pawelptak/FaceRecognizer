from kivy.clock import mainthread
from kivy.uix.screenmanager import Screen
from functions.configuration import *
import tkinter as tk
from tkinter import filedialog
from functions.face_detection import *
from functions.empty_dir import *
from concurrent.futures import ThreadPoolExecutor

class DetectionScreen(Screen):
    image_source = ''
    file_names = []
    detections_path = './detections/'
    number_detected: int = 0
    selected_index = 0

    def __init__(self, **kw):
        super().__init__(**kw)

    def detect(self):
        del_all_files(self.detections_path)

        self.number_detected = 0
        detected = 'Nothing'
        face_detector = load_face_detector()
        shape_predictor = load_shape_predictor()
        for index, file_name in enumerate(self.file_names):
            detected = face_detect(image_path=file_name, save_path=self.detections_path,
                                   face_name=self.ids.name_input.text, draw_points=True, face_det=face_detector,
                                   shape_pred=shape_predictor, file_number=index)
            self.file_names[index] = detected[0]
            self.number_detected += detected[1]
        # elif self.ids.cam_box.play:
        #    print('camera enabled')
        #    file_name = './detections/selfie.png'
        #    self.ids.cam_box.export_to_png(file_name)
        #    self.ids.camera_switch.trigger_action(duration=0.1)  # press button to turn off the camera
        #    self.ids.face_image.load_image(file_name)
        #    detected = face_detect(image_path=file_name, save_path=self.detections_path, face_name=self.ids.name_input.text, draw_points=True)

        self.show_results()

    @mainthread
    def show_results(self):
        if os.path.isfile(self.file_names[self.selected_index]):
            self.ids.face_image.source = self.file_names[self.selected_index]
            self.ids.face_image.reload()
        self.ids.result.text = 'Faces detected: ' + str(self.number_detected)
        self.ids.result.opacity = 1
        self.ids.detect_button.text = 'Detect'
        self.ids.detect_button.disabled = False

    def start_detection_thread(self):
        if self.ids.name_input.text != '' and len(self.file_names) > 0:
            self.ids.face_image.source = './Images/loading.gif'
            self.ids.result.opacity = 0
            self.ids.detect_button.text = 'Detecting'
            self.ids.detect_button.disabled = True
            executor = ThreadPoolExecutor(max_workers=1)
            executor.submit(self.detect)

    def open_file_dialog(self):
        self.selected_index = 0
        self.file_names = []
        root = tk.Tk()
        root.withdraw()
        file_names = filedialog.askopenfilenames(filetypes=[("Image files", ".jpeg .jpg .png .bmp .tiff")])

        self.get_root_window().raise_window()  # set focus on window

        if len(file_names) > 0:
            self.ids.num_loaded_text.opacity = 1
            self.ids.num_loaded_text.text = str(len(file_names)) + ' images loaded'
            for file_name in file_names:
                self.file_names.append(file_name)
            self.ids.face_image.load_image(self.file_names[self.selected_index])

    def load_image_source(self):
        img_source = load_config()
        return img_source

    def use_webcam(self):
        self.ids.num_loaded_text.opacity = 0
        self.ids.image_boxlayout.remove_widget(self.ids.face_image)
        if self.ids.camera_switch.text == 'OFF':
            self.ids.cam_box.play = True
            self.ids.face_image.reset_image()
            self.ids.cam_box.opacity = 1
            self.ids.image_button.disabled = True
        else:
            self.ids.image_button.disabled = False
            self.ids.cam_box.play = False
            self.ids.cam_box.opacity = 0

    def next_img(self):
        if self.selected_index < len(self.file_names) - 1:
            self.selected_index += 1
            self.ids.face_image.source = self.file_names[self.selected_index]
        print('index:', self.selected_index)

    def previous_img(self):
        if self.selected_index > 0:
            self.selected_index -= 1
            self.ids.face_image.source = self.file_names[self.selected_index]
        print('index:', self.selected_index)
