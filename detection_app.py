#Copyright (C) 2020 BardzoFajnyZespol

import threading

from kivy.app import App
from kivy.lang import Builder
from kivy.properties import ObjectProperty
from kivy.uix.gridlayout import GridLayout
from kivy.uix.screenmanager import ScreenManager
from kivy.core.window import Window
#importing classes. do not remove
from classes.detection_screen import DetectionScreen
from classes.drop_button import DropButton
from classes.screen_stack import ScreenStack

#loading ui files
Builder.load_file("ui files/widget_styles.kv")
Builder.load_file("ui files/navigation_ui.kv")
Builder.load_file("ui files/detection_screen.kv")



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
class FaceDetector(App):
    icon = 'Images/icon.png'

    Window.minimum_width, Window.minimum_height = (800,600)

    def build(self):
        # showing main screen
        return Main()

if __name__ == '__main__':
    FaceDetector().run()
