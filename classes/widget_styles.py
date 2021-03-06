from kivy.uix.button import Button


class CustButton(Button):
    pass


class RoundButton(Button):
    pass


class CustomSwitch(Button):
    on = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.text = 'OFF'

    def on_release(self):
        if not self.on:
            self.text = 'ON'
            self.on = True
        else:
            self.text = 'OFF'
            self.on = False
