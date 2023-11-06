"""Console script for caveat."""

import click
import yaml

from caveat.run import batch, run


@click.version_option(package_name="caveat")
@click.group()
def cli():
    """Console script for caveat."""
    pass


@cli.command(name="run")
@click.argument("config_path", type=click.Path(exists=True))
def run_command(config_path: click.Path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        run(config)


@cli.command(name="batch")
@click.argument("config_path", type=click.Path(exists=True))
def batch_command(config_path: click.Path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        batch(config)


# @cli.command(name="nrun")
# @click.argument("config_path", type=click.Path(exists=True))
# @click.option("--n", type=int, default=5)
# def nrun_command(config_path: click.Path):
#     with open(config_path, "r") as file:
#         config = yaml.safe_load(file)
