"""Console script for caveat."""

import click
import yaml

from caveat.run import (
    batch_command,
    nrun_command,
    nsample_command,
    report_command,
    run_command,
)


@click.version_option(package_name="caveat")
@click.group()
def cli():
    """Console script for caveat."""
    pass


@cli.command(name="run")
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--test", "-t", is_flag=True)
@click.option("--gen", "-g", is_flag=True)
@click.option("--verbose", "-v", is_flag=True)
def run(config_path: click.Path, test: bool, gen: bool, verbose: bool):
    """Train and report on an encoder and model as per the given configuration file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        run_command(config, verbose=verbose, test=test, gen=gen)


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--stats", "-s", is_flag=True)
def batch(config_path: click.Path, stats: bool):
    """Train and report on a batch of encoders and models as per the given configuration file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        batch_command(config, stats)


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--n", type=int, default=5)
@click.option("--stats", "-s", is_flag=True)
def nrun(config_path: click.Path, n: int, stats: bool):
    """Train and report variance on n identical runs with varying seeds."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        nrun_command(config, n, stats)


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--n", type=int, default=5)
@click.option("--stats", "-s", is_flag=True)
def nsample(config_path: click.Path, n: int, stats: bool):
    """Train and report variance on n identical runs with varying seeds."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        nsample_command(config, n, stats)


@cli.command()
@click.argument("observed_path", type=click.Path(exists=True))
@click.argument("logs_dir", type=click.Path(exists=True))
@click.option("--name", type=str, default="synthetic.csv")
@click.option("--verbose", is_flag=True)
@click.option("--head", type=int, default=10)
@click.option("--batch", "-b", is_flag=True)
def report(
    observed_path: click.Path,
    logs_dir: click.Path,
    name: str,
    verbose: bool,
    head: int,
    batch: bool,
):
    """Report on the given observed population and logs directory."""
    report_command(observed_path, logs_dir, name, verbose, head, batch)
