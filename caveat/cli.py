"""Console script for caveat."""

import click


@click.version_option(package_name="caveat")
@click.command()
def cli(args=None):
    """Console script for caveat."""
    click.echo("Replace this message by putting your code into caveat.cli.cli")
    click.echo("See click documentation at https://click.palletsprojects.com/")
    return 0
