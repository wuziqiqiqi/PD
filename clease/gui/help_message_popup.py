from kivy.uix.floatlayout import FloatLayout
from kivy.properties import ObjectProperty


class HelpMessagePopup(FloatLayout):
    close = ObjectProperty(None)

    def __init__(self, **kwargs):
        msg = kwargs.pop('msg')
        FloatLayout.__init__(self, **kwargs)
        self.ids.message.text = msg
