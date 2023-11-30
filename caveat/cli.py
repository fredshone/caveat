"""Console script for caveat."""

import click
import yaml

from caveat.run import batch_command, nrun_command, report_command, run_command


@click.version_option(package_name="caveat")
@click.group()
def cli():
    """Console script for caveat."""
    pass


@cli.command(name="run")
@click.argument("config_path", type=click.Path(exists=True))
def run(config_path: click.Path):
    """Train and report on an encoder and model as per the given configuration file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        run_command(config)


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
def batch(config_path: click.Path):
    """Train and report on a batch of encoders and models as per the given configuration file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        batch_command(config)


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--n", type=int, default=5)
def nrun(config_path: click.Path, n: int):
    """Train and report variance on n identical runs with varying seeds."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        nrun_command(config, n)


@cli.command()
@click.argument("observed_path", type=click.Path(exists=True))
@click.argument("logs_dir", type=click.Path(exists=True))
@click.option("--name", type=str, default="synthetic.csv")
@click.option("--verbose", is_flag=True)
@click.option("--head", type=int, default=10)
def report(
    observed_path: click.Path,
    logs_dir: click.Path,
    name: str,
    verbose: bool,
    head: int,
):
    """Report on the given observed population and logs directory."""
    report_command(observed_path, logs_dir, name, verbose, head)
