class ScreenStack:
    def __init__(self):
        self.stack = []

    def add_screen(self, screen_name):
        if len(self.stack) > 0:
            if self.stack[-1] != screen_name:
                self.stack.append(screen_name)
        else:
            self.stack.append(screen_name)

    def previous_screen(self):
        if len(self.stack) > 1:
            self.stack.pop()
        else:
            print("nothing to go back to")
        return self.stack[-1]

    def get_top(self):
        if len(self.stack) > 1:
            return self.stack[-1]
        else:
            return None