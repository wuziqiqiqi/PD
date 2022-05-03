from typing import Union, Sequence
from tkinter import TclError
from matplotlib import pyplot as plt
from ase.db import connect
from ase.gui.gui import GUI
from ase.gui.images import Images
import attr


@attr.define
class AnnotatedAx:
    """
    Container for collecting a matplotlib axis (ax) object along
    with annotations for each datapoint in the plot.

    ax: Axes instance
        Instance of the axes object containing the lines

    lines: Array of Line objects that annotations apply to

    annotations: Nested list with annotations
        Each list contains one annotation for each (x, y) pair on that
        line
    structure_names (Optional): Same shape as annotations, which contains the name of
        a structure for fetching data from the database.
    """

    ax = attr.field()
    lines = attr.field()
    annotations = attr.field()
    structure_names = attr.field(default=None)

    def __attrs_post_init__(self):
        for i, line in enumerate(self.lines):
            x, _ = line.get_data()
            if len(x) != len(self.annotations[i]):
                msg = f"Annotations for line {i} "
                msg += f"has length {len(self.annotations[i])} "
                msg += f"but there are {len(x)} data points!"
                raise ValueError(msg)


class InteractivePlot:
    """Interactive plot with annotations.

    Parameters:

    fig: Figure instance
        Instance of the figure object visualizing the data

    annotated_axes: Sequence of AnnotatedAx, which contains information
        on the annotations for lines in the ax object.
    """

    def __init__(self, fig, annotated_axes: Union[AnnotatedAx, Sequence[AnnotatedAx]]):
        self.fig = fig
        self.annotated_axes = annotated_axes

        self.all_annotations = []
        for annot_ax in self.annotated_axes:
            annot = annot_ax.ax.annotate(
                "",
                xy=(0, 0),
                xytext=(-20, 20),
                textcoords="offset points",
                bbox=dict(boxstyle="round", fc="w"),
                arrowprops=dict(arrowstyle="->"),
            )
            annot.set_visible(False)
            self.all_annotations.append(annot)
        assert len(self.all_annotations) == len(self.annotated_axes)
        # Initialize the active annotation to the first annotation object.
        # This reference may be moved to a different axis later.
        self.active_annot = self.all_annotations[0]
        self.active_line_index = 0

        self.fig.canvas.mpl_connect("motion_notify_event", self.hover)
        # It seems we need to show now, otherwise the interactivity is lost?
        # TODO: Figure out if this can be avoided.
        plt.show()

    @property
    def annotated_axes(self) -> Sequence[AnnotatedAx]:
        return self._annotated_axes

    @annotated_axes.setter
    def annotated_axes(self, value) -> None:
        if isinstance(value, AnnotatedAx):
            # It's a single AnnotatedAx object, put it in a list
            # with itself, so the rest of the logic fits.
            self._annotated_axes = [value]
            return
        # Assume it's a list-like type of AnnotatedAx objects
        # Verify that each one is of the correct type.
        for annot_ax in value:
            if not isinstance(annot_ax, AnnotatedAx):
                raise TypeError(f"Each annotated axis object must be AnnotatedAx, got {annot_ax!r}")
        self._annotated_axes = value

    def _update_annotation(self, ind, event_index: int):
        """Update the annotation shown."""
        ax_obj = self.annotated_axes[event_index]
        lines = ax_obj.lines
        line = lines[self.active_line_index]
        x, y = line.get_data()
        self.active_annot = self.all_annotations[event_index]
        self.active_annot.xy = (x[ind["ind"][0]], y[ind["ind"][0]])
        anot = ax_obj.annotations[self.active_line_index]
        text = anot[ind["ind"][0]]
        self.active_annot.set_text(text)

    def hover(self, event):
        """React on a hover event."""
        vis = self.active_annot.get_visible()
        if self.event_in_ax(event):
            event_index = self.get_event_index(event)
            cont = False
            lines = self.annotated_axes[event_index].lines
            for i, line in enumerate(lines):
                cont, ind = line.contains(event)
                if cont:
                    self.active_line_index = i
                    break

            if cont:
                self._update_annotation(ind, event_index)
                self.active_annot.set_visible(True)
                self.fig.canvas.draw_idle()
            else:
                if vis:
                    self.active_annot.set_visible(False)
                    self.fig.canvas.draw_idle()

    def event_in_ax(self, event) -> bool:
        """Is an event within an active axes?"""
        return any(event.inaxes == annot_ax.ax for annot_ax in self.annotated_axes)

    def get_event_index(self, event) -> int:
        """Get the index of the axes object corresponding to the event."""
        for ii, annot_ax in enumerate(self.annotated_axes):
            if annot_ax.ax == event.inaxes:
                return ii
        raise RuntimeError(f"Didn't find index corresponding to event {event}.")

    def get_event_ax(self, event) -> AnnotatedAx:
        """Retrieve the AnnotatedAx object which corresponds to an event."""
        idx = self.get_event_index(event)
        return self.annotated_axes[idx]


class ShowStructureOnClick(InteractivePlot):
    def __init__(self, fig, axes: Union[AnnotatedAx, Sequence[AnnotatedAx]], db_name: str):
        self.db_name = db_name
        self.active_images = Images()
        # Uninitilized GUI. We don't create this until we click,
        # since we otherwise just open a new empty GUI window.
        self.gui = None
        fig.canvas.mpl_connect("button_press_event", self.on_click)
        super().__init__(fig, axes)

    def on_click(self, event) -> None:

        if not self.event_in_ax(event):
            return

        if event.button == 1:
            # Find the index of the point
            annot_ax = self.get_event_ax(event)

            for i, line in enumerate(annot_ax.lines):
                is_contained, ind = line.contains(event)
                if is_contained:
                    self.active_line_index = i
                    break
            else:
                # We clicked on something which wasn't a datapoint
                # because we never broke out of the loop.
                return

            # Prepare to grab the atoms object from the data base.
            db = connect(self.db_name)
            atoms = []
            self._close_gui()

            if annot_ax.structure_names is not None:
                name = annot_ax.structure_names[self.active_line_index][ind["ind"][0]]
            else:
                # Try falling back on the annotations as name (old)
                name = annot_ax.annotations[self.active_line_index][ind["ind"][0]]

            for row in db.select(name=name):
                atoms.append(row.toatoms())
            if not atoms:
                # Didn't find anything.
                return
            self.active_images.initialize(atoms)
            # Create a new GUI instance and run it
            self.gui = GUI(self.active_images)
            self.gui.run()

    def _close_gui(self) -> None:
        if self.gui is not None:
            # Try to close the existing GUI, if we already opened one
            try:
                self.gui.exit()
            except TclError:
                pass
