from clease.gui.legend_item import LegendItem


class Legend(object):

    def __init__(self, layout):
        self.layout = layout

    def setup(self, new_items, num_rows=1, num_cols=1):
        """
        Initialise the legend

        Parameters:

        items: List of dict
            List of dictionaries for each item. The dictionary
            constain {'text': legend_text, 'color': rgb color}

        num_rows: int
            Number of rows

        num_cols: int
            Number of columns
        """
        self.layout.clear_widgets()
        if num_rows * num_cols < len(new_items):
            raise ValueError('num_rows*num_cols < len(items)!')

        counter = 0
        for r in range(num_rows):
            for c in range(num_cols):
                item = LegendItem(icon_color=new_items[counter]['color'],
                                  text=new_items[counter]['text'],
                                  icon_size=[20, 20])
                self.layout.add_widget(item)
                counter += 1
                if counter >= len(new_items):
                    return
