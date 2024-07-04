#!/usr/bin/env python
from pathlib import Path, PurePath
import click
from clease_cxx import has_parallel
import clease
from clease.version import __version__


@click.group()
@click.version_option(__version__)
def clease_cli():
    """The main CLEASE CLI"""


@clease_cli.command()
def info():
    """Print some information about the current CLEASE installation."""
    is_par = has_parallel()
    inst_path = get_install_path()
    s = [f"Version: {__version__}"]
    s += [f"C++ OpenMP: {is_par}"]
    s += [f"Install path: {inst_path}"]
    click.echo("\n".join(s))


def get_install_path() -> PurePath:
    """Get the location of the CLEASE installation"""
    return (Path(clease.__file__).parent).resolve()
