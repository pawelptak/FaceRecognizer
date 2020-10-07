from kivy.uix.screenmanager import Screen
from functions.haar_cascade import *
from functions.hog import *
from functions.configuration import *
from functions.crop_image import *
import tkinter as tk
from tkinter import filedialog

class HAARScreen(Screen):
    image_source = ''

    def __init__(self, **kw):
        super().__init__(**kw)

    def detect(self):
        if self.ids.face_image.image_loaded or self.ids.cam_box.play:
            self.haar_detect()

    def haar_detect(self):
        if not self.ids.cam_box.play:
            self.ids.face_image.source = self.ids.face_image.original_source
            self.ids.face_image.reload()
        else:
            print('camera enabled')
            file_name = './detections/selfie.png'
            self.ids.cam_box.export_to_png(file_name)
            self.ids.camera_switch.trigger_action(duration=0.1) #press button to turn off the camera
            self.ids.face_image.load_image(file_name)

        img = self.ids.face_image.source
        img_name = os.path.basename(img)

        min_size = self.ids.setting_3_input.text

        #HAAR detection
        #detection = face_detect_haar(img, float(self.ids.setting_1_input.text), int(self.ids.setting_2_input.text), tuple(map(int, min_size.split(' '))))

        #HOG detection
        detection = face_detect_hog(img)
        #detection = face_detect_haar(img, 1.5, 5, (10,10))
        self.ids.result.opacity = 1

        if len(detection) is not 0:
            cv2_image = cv2.imread(img)
            save_path = './detections/' + img_name
            cv2.imwrite(save_path, cv2_image) #saving image without detections
            face_num = 1
            for (x, y, w, h) in detection:
                cv2_image = cv2.rectangle(cv2_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cropped = crop_image(img, x, y, w, h)
                save_path = './detections/face_'+str(face_num)+'_' + img_name
                face_num += 1
                cv2.imwrite(save_path, cropped)  # saving detected faces as images
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
        file_name = filedialog.askopenfilename(filetypes=[("Image files", ".jpeg .jpg .png .bmp .tiff")])

        self.get_root_window().raise_window()  # set focus on window


        if file_name != '':
            # self.ids.face_image.source = file_name
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

    def use_webcam(self):
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
