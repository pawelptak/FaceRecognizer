


import threading
from functions.configuration import *
from kivy.app import App
from kivy.lang import Builder
from kivy.properties import ObjectProperty
from kivy.uix.gridlayout import GridLayout
from kivy.uix.screenmanager import ScreenManager
from kivy.core.window import Window
#importing classes. do not remove
from classes.detection_screen import DetectionScreen
from classes.training_screen import TrainingScreen
from classes.drop_button import DropButton
from classes.face_image import FaceImage
from classes.screen_stack import ScreenStack
from functions.empty_dir import *

Builder.load_file("ui files/widget_styles.kv")
Builder.load_file("ui files/navigation_ui.kv")
Builder.load_file("ui files/detection_screen.kv")
Builder.load_file("ui files/training_screen.kv")

#Main Screen with navigation bar on top
class Main(GridLayout, threading.Thread):
    manager = ObjectProperty(None)



#manager for changing screens
class WindowManager(ScreenManager):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.stack = ScreenStack()
        self.stack.add_screen("detection")

#main app class
class FaceRecognizer(App):
    # loading ui files

    icon = 'Images/icon.png'

    Window.minimum_width, Window.minimum_height = (800,600)

    def build(self):
        # showing main screen
        return Main()

if __name__ == '__main__':

    if not os.path.exists('./detections'):
        os.makedirs('./detections')
    del_all_files('./detections')
    clear_image_src()

    FaceRecognizer().run()
    del_all_files('./detections')
