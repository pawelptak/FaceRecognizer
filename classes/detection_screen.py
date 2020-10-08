from kivy.uix.screenmanager import Screen
from functions.haar_cascade import *
from functions.hog import *
from functions.configuration import *
from functions.crop_image import *
from functions.face_alignment import *
import tkinter as tk
from tkinter import filedialog

class DetectionScreen(Screen):
    image_source = ''
    file_names = ''

    def __init__(self, **kw):
        super().__init__(**kw)

    def detect(self):
        if self.ids.name_input.text != '':
            if self.ids.face_image.image_loaded:
                for image in self.file_names:
                    self.haar_detect(image)
            elif self.ids.cam_box.play:
                print('camera enabled')
                file_name = './detections/selfie.png'
                self.ids.cam_box.export_to_png(file_name)
                self.ids.camera_switch.trigger_action(duration=0.1)  # press button to turn off the camera
                self.ids.face_image.load_image(file_name)
                self.haar_detect(file_name)

    def haar_detect(self, image):

        self.ids.face_image.source = image
        self.ids.face_image.reload()


        img = self.ids.face_image.source
        img_name = os.path.basename(img)



        #HAAR detection
        #detection = face_detect_haar(img, 1.5, 5, (10,10))

        #HOG detection
        detection = face_detect_hog(img)

        self.ids.result.opacity = 1

        if len(detection) is not 0:
            cv2_image = cv2.imread(img)
            save_path = './detections/' + img_name
            cv2.imwrite(save_path, cv2_image) #saving image without detections

            #for (x, y, w, h) in detection:
                #cv2_image = cv2.rectangle(cv2_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

            for f in detection:
                align_face(cv2_image, f, self.ids.name_input.text)
                cv2_image = cv2.rectangle(cv2_image, (f.left(), f.top()), (f.right(), f.bottom()), (255, 0, 0), 2)

            save_path = './detections/det_' + img_name
            cv2.imwrite(save_path, cv2_image) #saving image with detections


            self.ids.face_image.source = save_path
            self.ids.face_image.reload()
            self.ids.result.text = 'Faces detected: ' + str(len(detection))
        else:
            self.ids.result.text = 'No faces detected'

    def open_file_dialog(self):

        root = tk.Tk()
        root.withdraw()
        file_names = filedialog.askopenfilenames(filetypes=[("Image files", ".jpeg .jpg .png .bmp .tiff")])

        self.get_root_window().raise_window()  # set focus on window

        self.file_names = file_names
        self.ids.face_image.load_image(self.file_names[0])
        self.ids.num_loaded_text.opacity = 1
        self.ids.num_loaded_text.text = str(len(self.file_names)) + ' loaded'


    def load_image_source(self):
        img_source = load_config()
        return img_source

    def on_pre_enter(self, *args):
        src = load_config()
        if src:
            self.ids.face_image.load_image(src)
            self.ids.result.opacity = 0

    def use_webcam(self):
        self.ids.num_loaded_text.opacity = 0
        self.ids.image_boxlayout.remove_widget(self.ids.face_image)
        if self.ids.camera_switch.text == 'OFF':
            self.ids.cam_box.play = True
            self.ids.face_image.reset_image()
            self.ids.cam_box.opacity = 1
            self.ids.image_button.disabled = True
        else:
            self.ids.cam_box.play = False
            self.ids.cam_box.opacity = 0
            self.ids.image_button.disabled = False
