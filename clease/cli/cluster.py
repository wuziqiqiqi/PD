import click
from clease.settings import settings_from_json
from . import main


@main.clease_cli.command()
@click.argument(
    "filename",
    type=click.Path(exists=True),
)
@click.option("--gui", "-g", is_flag=True, help="Open clusters in an ASE GUI?")
@click.option(
    "--no-table",
    "-t",
    is_flag=True,
    default=False,
    help="Disable printing the cluster table.",
)
def clusters(filename, gui, no_table):
    """Print the clusters that a settings object (saved as a JSON file) produces.

    FILENAME is the name of the settings JSON file
    """
    settings = settings_from_json(filename)
    if no_table is False:
        table = settings.clusters_table()
        click.echo(table)

    if gui:
        # We use ase.visualize.view here, to have the GUI spawned in
        # a background process, rather than using the settings.view_clusters()
        # method.
        # pylint: disable=import-outside-toplevel
        from ase.visualize import view

        figures = settings.get_all_figures_as_atoms()
        view(figures)
