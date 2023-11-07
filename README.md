<!--- the "--8<--" html comments define what part of the README to add to the index page of the documentation -->
<!--- --8<-- [start:docs] -->
![caveat](resources/logos/title.png)

Generative models for human activity sequences.

[![Daily CI Build](https://github.com/fredshone/caveat/actions/workflows/daily-scheduled-ci.yml/badge.svg)](https://github.com/fredshone/caveat/actions/workflows/daily-scheduled-ci.yml)
[![Documentation](https://github.com/fredshone/caveat/actions/workflows/pages/pages-build-deployment/badge.svg)](https://fredshone.github.io/caveat)

Caveat is for building models that generate human activity sequences.

## Framework

Caveat provides a framework to train and test generative models. A model run is composed of:

- Data - see the caveat examples for synthetic and real data generation
- Encoder - see caveat encoders for available encoders
- Model - see caveat models for available models
- Report - see caveat report

## Quick Start

Once installed get started using `caveat --help`.

`caveat run --help`

Train and report on a model using `caveat run configs/toy_run.yaml`. The run data, encoder, model and other parameters are controlled using a run config. This will write results and tensorboard logs to `logs/` (this is configurable). Monitor or review training progress using `tensorboard --logdir=logs`.

`caveat batch --help`

Train and report on a batch of runs using a special batch config `caveat batch configs/toy_batch.yaml`. Batch allows comparison of multiple models and/or hyper-params as per the batch config.

`caveat nrun --help`

Run and report the variance of n of the same run using `caveat nrun configs/toy_run.yaml --n 3`. The config is as per a regular run config but `seed` is ignored.

## Data

Caveat requires a .csv format to represent a *population* of *activity sequences*:

| pid | act | start | end |
|---|---|---|---|
| 0 | home | 0 | 390 |
| 0 | work | 390 | 960 |
| 0 | home | 960 | 1440 |
| 1 | home | 0 | 390 |
| 1 | education | 390 | 960 |
| 1 | home | 960 | 1440 |

- **pid** (person id) field is a unique identifier for each sequence
- **act** is a categorical value for the type of activity in the sequence
- **start** and **end** are the start and end times of the activities in the sequence

We commonly refer to these as ***populations***. Times are assumed to be in minutes and should be integers. Valid sequences should be complete, ie the start of an activity should be equal to the end of the previous. The convention is to start at midnight. Such that time can be thought of as *minutes since midnight*.

There is an example [toy population](https://github.com/fredshone/caveat/latest/examples/data) with 1000 sequences in `caveat/examples/data`. There are also [example notebooks](https://github.com/fredshone/caveat/tree/main/examples) for:

- Generation of a synthetic population
- Generation of a population from UK travel diaries

### Encoder

We are keen to test different encodings (such as continuous sequences versus descretised time-steps). The exact encoding required will depend on the model structure being used.

The encoder and it's parameters are defined in the config `encoder` group.

Encoders are defined in the `encoders` module and should be accessed via `caveat.encoders.library`.

Note that encoders must implement both an encode and decode method so that model outputs can be converted back into the population format for reporting.

## Model

The model and it's parameters are defined in the config `model` group. Models are trained until validation stabilises or until some max number of epochs.

Models are defined in `models` and should be accessed via `caveat.models.library`. Models and their training should be specified via the config.

The `data_loader`, `experiment` and `trainer` hyper-params are also configured by similarly named groups. These groups use the standard [pytorch-lightning](https://pypi.org/project/pytorch-lightning/) framework.

## Report

Each model (with training weights from the best performing validation step) is used to generate a new "sythetic" population.

Sythetic populations are compared to the original "observed" population.

Reporting the quality of generated populations is subjective. The `features` module provides functions for extracting features from populations. Such as "average activity durations". These are then used to make comparison metrics between the observed and sythetic populations.

<!--- --8<-- [end:docs] -->

## Documentation

For more detailed instructions, see our [documentation](https://fredshone.github.io/caveat/latest).

## Installation

To install caveat, we recommend using the [mamba](https://mamba.readthedocs.io/en/latest/index.html) package manager:

### As a user
<!--- --8<-- [start:docs-install-user] -->


``` shell

mamba create -n caveat -c conda-forge -c city-modelling-lab -c pytorch

```
<!--- --8<-- [end:docs-install-user] -->

### As a developer
<!--- --8<-- [start:docs-install-dev] -->
``` shell
git clone git@github.com:fredshone/caveat.git
cd caveat
mamba create -n caveat -c conda-forge -c city-modelling-lab -c pytorch --file requirements/base.txt --file requirements/dev.txt
mamba activate caveat
pip install --no-deps -e .
```
<!--- --8<-- [end:docs-install-dev] -->
For more detailed instructions, see our [documentation](https://fredshone.github.io/caveat/latest/installation/).

## Contributing

There are many ways to contribute to caveat.
Before making contributions to the caveat source code, see our contribution guidelines and follow the [development install instructions](#as-a-developer).

If you plan to make changes to the code then please make regular use of the following tools to verify the codebase while you work:

- `pre-commit`: run `pre-commit install` in your command line to load inbuilt checks that will run every time you commit your changes.
The checks are: 1. check no large files have been staged, 2. lint python files for major errors, 3. format python files to conform with the [pep8 standard](https://peps.python.org/pep-0008/).
You can also run these checks yourself at any time to ensure staged changes are clean by simple calling `pre-commit`.
- `pytest` - run the unit test suite and check test coverage.
- `pytest -p memray -m "high_mem" --no-cov` (not available on Windows) - after installing memray (`mamba install memray pytest-memray`), test that memory and time performance does not exceed benchmarks.

For more information, see our [documentation](https://fredshone.github.io/caveat/latest/contributing/).

## Building the documentation

If you are unable to access the online documentation, you can build the documentation locally.
First, [install a development environment of caveat](https://fredshone.github.io/caveat/latest/contributing/coding/), then deploy the documentation using [mike](https://github.com/jimporter/mike):

```
mike deploy develop
mike serve
```

Then you can view the documentation in a browser at http://localhost:8000/.


## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [arup-group/cookiecutter-pypackage](https://github.com/arup-group/cookiecutter-pypackage) project template.