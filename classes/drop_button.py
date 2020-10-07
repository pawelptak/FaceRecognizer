from classes.widget_styles import RoundButton
from kivy.core.window import Window
from pathlib import Path

#button to drag and drop files on
class DropButton(RoundButton):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Window.bind(on_dropfile=self.on_file_drop)
        self.path = ''


    def on_file_drop(self, window, file_path):
        path = file_path.decode("utf-8")
        path = Path(path)

        #print('box position', str(self.pos))
        #print('box size', str(self.size))
        #print('mouse position', window.mouse_pos)
        within_box_width = window.mouse_pos[0]>=self.pos[0] and window.mouse_pos[0] <= self.pos[0] + self.size[0]
        within_box_height = window.mouse_pos[1]>=self.pos[1] and window.mouse_pos[1] <= self.pos[1] + self.size[1]

        if within_box_width and within_box_height:
            self.path = path
            print(self.path)
            self.children[1].load_image(str(self.path))