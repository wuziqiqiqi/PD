#!/usr/bin/env python
import click
import clease


@click.group()
@click.version_option(clease.__version__)
def clease_cli():
    """The main CLEASE CLI"""
