#!/usr/bin/env python
import os
import argparse
from .db import db_cli


# For Windows dependencies see
# https://stackoverflow.com/questions/40769386/kivy-windows-unable-to-find-any-valuable-window-provider-at-all
def install_gui_dependencies(args):
    import subprocess
    subprocess.check_call(['pip', 'install', 'kivy>=1.11,<2'] + args)

    if os.name == 'nt':
        # Windows dependencies
        pckgs = 'docutils pygments pypiwin32 kivy.deps.sdl2 kivy.deps.glew'
        subprocess.check_call(['pip', 'install'] + pckgs.split() + args)

    url = 'https://github.com/kivy-garden/graph/archive/master.zip'
    subprocess.check_call(['pip', 'install', url] + args)


def main():
    parser = argparse.ArgumentParser(description="CLEASE CLI")
    subparsers = parser.add_subparsers(help="Sub command help", dest="command")
    gui_parser = subparsers.add_parser("gui", help="Launches the CLEASE GUI")
    gui_parser.add_argument("--setup",
                            help="Install missing dependencies required to launch the "
                            "GUI",
                            action="store_true")

    db_parser = subparsers.add_parser("db", help="Launch the CLEASE DB CLI")
    db_parser.add_argument("name", help="Name of the database")
    db_parser.add_argument("--show",
                           help="[tab, names, cf]. If tab: The name of all correlation "
                           "function tables is shown.\n"
                           "If names, the name of all known correlation functions "
                           "is shown\n"
                           "If cf, the correlation functions of the given ID is "
                           "shown (See ID argument)")
    db_parser.add_argument("--id", help="Database ID to operate on")
    args = parser.parse_args()
    if args.command == 'gui':
        if args.setup:
            install_gui_dependencies([])
        else:
            from clease.gui.clease_gui import CleaseGUI
            CleaseGUI().run()
    elif args.command == 'db':
        db_cli(args)


if __name__ == "__main__":
    main()
