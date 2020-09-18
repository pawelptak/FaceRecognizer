from kivy.uix.screenmanager import Screen
from functions.haar_cascade import *

class DetectionScreen(Screen):

	def __init__(self, **kw):
		super().__init__(**kw)

	def haar_detect(self):
		img = self.ids.face_image.source
		detection = face_detect_haar(img)
		if detection is not None:
			self.ids.face_image.source = detection


