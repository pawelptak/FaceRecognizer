from kivy.uix.image import Image
from functions.configuration import *

class FaceImage(Image):
    image_loaded = False
    original_source = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_image(self, img_src: str):
        self.image_loaded = True
        self.original_source = img_src
        self.source = img_src
        self.reload()

    def reset_image(self):
        self.image_loaded = False
        self.original_source = None
        self.source = './Images/image.png'
