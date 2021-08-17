import os
import subprocess
import click
from . import main


# For Windows dependencies see
# https://stackoverflow.com/questions/40769386/kivy-windows-unable-to-find-any-valuable-window-provider-at-all
def install_gui_dependencies(args):
    subprocess.check_call(['pip', 'install', 'kivy>=1.11,<2'] + args)

    if os.name == 'nt':
        # Windows dependencies
        pckgs = 'docutils pygments pypiwin32 kivy.deps.sdl2 kivy.deps.glew'
        subprocess.check_call(['pip', 'install'] + pckgs.split() + args)

    url = 'https://github.com/kivy-garden/graph/archive/master.zip'
    subprocess.check_call(['pip', 'install', url] + args)


@main.clease_cli.command(help="Launches the CLEASE GUI")
@click.option('--setup',
              is_flag=True,
              help="Install missing dependencies required to launch the GUI")
def gui(setup):
    if setup:
        install_gui_dependencies([])
    else:
        from clease.gui.clease_gui import CleaseGUI
        CleaseGUI().run()
