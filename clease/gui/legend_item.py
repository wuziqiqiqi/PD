from kivy.uix.widget import Widget
from kivy.properties import ListProperty, StringProperty


class LegendItem(Widget):
    icon_color = ListProperty([1, 0, 0])
    icon_size = ListProperty([1, 1])
    text_color = ListProperty([1, 1, 1])
    text = StringProperty('sample')
