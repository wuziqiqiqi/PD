import click
from clease.settings import settings_from_json
from clease import tools
from . import main


@main.clease_cli.command()
@click.argument(
    "filename",
    type=click.Path(exists=True),
)
def reconfigure(filename):
    """
    Reconfigure a settings file saved as a JSON file.
    Will reconfigure the database that the settings points to.

    FILENAME is the name of the settings JSON file
    """
    settings = settings_from_json(filename)
    tools.reconfigure(settings, verbose=True)
