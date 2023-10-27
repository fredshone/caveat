"""Console script for caveat."""

import click
import yaml

from caveat.run import runner


@click.version_option(package_name="caveat")
@click.group()
def cli():
    """Console script for caveat."""
    pass


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
def run(config_path: click.Path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        runner(config)
