"""Console script for caveat."""

import click
import yaml

from caveat.run import batch, nrun, run


@click.version_option(package_name="caveat")
@click.group()
def cli():
    """Console script for caveat."""
    pass


@cli.command(name="run")
@click.argument("config_path", type=click.Path(exists=True))
def run_command(config_path: click.Path):
    """Train and report on an encoder and model as per the given configuration file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        run(config)


@cli.command(name="batch")
@click.argument("config_path", type=click.Path(exists=True))
def batch_command(config_path: click.Path):
    """Train and report on a batch of encoders and models as per the given configuration file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        batch(config)


@cli.command(name="nrun")
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--n", type=int, default=5)
def nrun_command(config_path: click.Path, n: int):
    """Train and report variance on n identical runs with varying seeds."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        nrun(config, n)
