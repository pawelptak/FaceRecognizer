from kivy.uix.image import Image


class FaceImage(Image):
    image_loaded = False
    original_source = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_image(self, img_src: str):
        self.image_loaded = True
        self.original_source = img_src
