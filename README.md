<!--- the "--8<--" html comments define what part of the README to add to the index page of the documentation -->
<!--- --8<-- [start:docs] -->
![caveat](resources/logos/title.png)

# CAVEAT

Generative models for human activity sequences.

[![Daily CI Build](https://github.com/fredshone/caveat/actions/workflows/daily-scheduled-ci.yml/badge.svg)](https://github.com/fredshone/caveat/actions/workflows/daily-scheduled-ci.yml)
[![Documentation](https://github.com/fredshone/caveat/actions/workflows/pages/pages-build-deployment/badge.svg)](https://fredshone.github.io/caveat)

Caveat is for building a comparing models that generate human activity sequences. This includes:

- Methods for generating training datasets, either:
  - Synthetic data
  - UK National Travel Survey data
- A framework for training models
- Metrics for comparing models

## Training Data

Caveat uses a simple .csv format to represent a population of activity sequences, we commonly refer to these as *populations*:

| pid | act | start | end |
|---|---|---|---|
| 0 | home | 0 | 390 |
| 0 | work | 390 | 960 |
| 0 | home | 960 | 1440 |
| 1 | home | 0 | 390 |
| 1 | education | 390 | 960 |
| 1 | home | 960 | 1440 |

- **pid** (Person id) field is a unique identifier for each sequence
- **act** is a categorical value for the type of activity in the sequence
- **start** and **end** are the start and end of the activities in the sequence

Times are assumed to be in minutes and should be integers.

Valid sequences should be complete, ie the start of an activity should be equal to the end of the previous. The convention is to start at midnight. Such that time can be thought of as *minutes since midnight*.

There is an example toy population with 1000 sequences in the [examples](https://github.com/fredshone/caveat/latest/examples/data). There are also notebooks for:

- [Generation of a synthetic population](https://fredshone.github.io/caveat/latest/examples/1_synthetic_population_generation.ipynb)
- [Generation of a population from travel diaries](https://fredshone.github.io/caveat/latest/examples/2_NTS_population_generation.ipynb)

## Training and Testing Models

Once you have generated a population of activity sequences you can train generative models and test how well they are able to recreate the original population. Testing uses metrics to quantify how good the generated populations are. Metrics are covered in the next section.

A typical experiment trains a model on some given **population**, using a sequence **encoding** and **model** structure. The trained model is then used to generate a new population which can be compared to the original using metrics.

To facilitate rapid reproducible experiemnts, the specification of the **population**, **encoding** and **model** are orchestrated via [config files](https://fredshone.github.io/caveat/latest/configs/).

As an example (from the project root) you can run the toy synthetic population through a simple VAE model using `caveat run configs/vae-toy.yaml`. This will write results and tensorboard logs to `logs/`.

#### Tensorboard

Monitor or review model training progress using `tensorboard --logdir=SAVE_DIR`. Where the `SAVE_DIR` is specified in the config (default is `logs`).

### Model Library

Models are defined in `models` and should be accessed via `caveat.models.library`. Models and their training should be specified via the config.

### Encoding Library

We are keen to test different encodings (such as continuous sequences versus descretised time-steps). The exact encoding required will depend on the model structure being used.

Encoders are defined in `encoders` and should be accessed via `caveat.encoders.library`. Note that encoders must implement both an encode and decode method so that model outputs can be converted back into the population of activity sequences format.

## Metrics

The `metrics` module provides measures of how well a generated population represents it's original "observed" population.

Metrics are boradly intended to measure the distance of distributions between two populations. Such that models and their associated encodings can be systematically tested.

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