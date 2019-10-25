#!/usr/bin/env python
import sys
import os


def print_help_msg():
    msg = "==================================================\n"
    msg += "'clease' command requires additional argument(s).\n"
    msg += "Currently, allowed arguments are:\n"
    msg += "gui : launch GUI for CLEASE\n"
    msg += "gui-setup : Install additional dependencies needed\n"
    msg += "            for running GUI.\n"
    msg += "==================================================\n"
    print(msg)


# For Windows dependencies see
# https://stackoverflow.com/questions/40769386/kivy-windows-unable-to-find-any-valuable-window-provider-at-all
def install_gui_dependencies(args):
    import subprocess
    subprocess.check_call(['pip', 'install', 'kivy'] + args)

    if os.name == 'nt':
        # Windows dependencies
        pckgs = 'docutils pygments pypiwin32 kivy.deps.sdl2 kivy.deps.glew'
        subprocess.check_call(['pip', 'install'] + pckgs.split() + args)

    url = 'https://github.com/kivy-garden/graph/archive/master.zip'
    subprocess.check_call(['pip', 'install', url] + args)


def main():
    argv = sys.argv
    num_args = len(argv)
    if num_args == 1:
        print_help_msg()
        return

    if argv[1] == 'gui':
        from clease.gui.cleaseGUI import CleaseGUI
        CleaseGUI().run()

    elif argv[1] == 'gui-setup':
        install_gui_dependencies(argv[2:])

    elif argv[1] in ['-help', '-h', '--help', 'help']:
        print_help_msg()
        return

if __name__ == "__main__":
    main()
