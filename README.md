<!--- the "--8<--" html comments define what part of the README to add to the index page of the documentation -->
<!--- --8<-- [start:docs] -->
![caveat](resources/logos/title.png)

Generative models for human schedules.

[![Daily CI Build](https://github.com/fredshone/caveat/actions/workflows/daily-scheduled-ci.yml/badge.svg)](https://github.com/fredshone/caveat/actions/workflows/daily-scheduled-ci.yml)
[![Documentation](https://github.com/fredshone/caveat/actions/workflows/pages/pages-build-deployment/badge.svg)](https://fredshone.github.io/caveat)

Caveat is for building models that generate human activity schedules. Activity scheduling is required component of activity-based models. There are applocations for modelling transport, energy and epidemiological systems.

## Overview

Caveat provides a framework to train and evaluate generative models. A Caveat model is composed of the following modules:

- [Data](#data) - loaders and augmentation for tabular input data 
- [Encoding](#encoding) - encoding/decoding inputs into various data representations
- [Model](#model) - generative models to learn and synthesize new populations of sequences
- Experiment - the parameterisation and operation of model training
- [Evaluate](#evaluate) - measuring the quality of synthetic populations of sequences

Caveat is designed to be installed and run as a command line application. The above modules are controlled using config files such that experiments are accessible to new users and can be reproduced. In particular we provide the `caveat batch` command to allow systematic comparison of different models or model specifications.

## Motivation

A core requiremnt of caveat is to allow implementation, training and evaluation of different generative models. A core requirement of such models is to be able to generate samples that are realistic individually (ie are structurally valid) and in aggregate.

Caveat can additionally be used for training and evaluating conditional generative models, where the requirement is extended to the generation of samples that are realistic individually (ie are structurally valid) and in aggregate with respect to the attributes of each individual.

## Quick Start

Once [installed](#installation) get started using `caveat --help`.

Caveat uses the following commands to run experiments:

### Run

`caveat run --help`

Train and evaluate a single model. The run data, encoder, model and other parameters are controlled using a run config. For example - `caveat run configs/toy_run.yaml` or `caveat run configs/toy_run_conditional.yaml` for a conditional model.

This will write an output synthetic population, any batch outputs, model checkpoints and evaulation to the (default) `logs` directory.

### Batch

`caveat batch --help`

Train and report on a batch of runs using a batch config. Batch allows comparison of multiple models and/or hyper-params as per the batch config. For example - `caveat batch configs/toy_batch.yaml` or `caveat batch configs/toy_batch_conditional.yaml` for a conditional model.

### Nrun

`caveat nrun --help`

Nrun is a simplified version of batch used to repeat the same model training and evaluation. This is intended to test for variance in model training and sampling. For example, run and evaluate the variance of _n=3_ of the same run using `caveat nrun configs/toy_run.yaml --n 3`. The config is as per a regular run config but `seed` is ignored.

### Nsample

`caveat nsample --help`

As per nrun but only assesses variance from the sampling process (not model training).

### Reporting

`caveat report --help`

Evaluate the outputs of an existing run or batch (using `-b`).

### Logging

Caveat writes tensorboard logs to a (default) `logs/` directory. Monitor or review training progress using `tensorboard --logdir=logs`.

## Data

Caveat expects inputs as .csv format with the following headers.

### Activity sequences

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

Times are assumed to be in minutes and should be integers. Valid sequences should be complete, ie the start of an activity should be equal to the end of the previous. The convention is to start at midnight. Such that time can be thought of as *minutes since midnight*.

### Attributes

Caveat supports conditional generation, for which individual input sequences also require attributes.

| pid | gender | age | employment |
|---|---|---|---|
| 0 | F | 24 | FTW |
| 1 | M | 85 | NEET |

- **pid** (person id) field is a unique identifier dsignating the attributes for each above sequence

Other than the `pid` column, columns can be arbitrarilly named and represented with any data type.

### Examples

There are example [toy populations](https://github.com/fredshone/caveat/latest/examples/data) with 1000 sequences in `caveat/examples/data`. There are also [example notebooks](https://github.com/fredshone/caveat/tree/main/examples) for:

- Generation of a synthetic population
- Generation of a population from UK travel diaries (requires access to UK NTS trip data)

## Encoding

### Schedules

Input schedules are configured as follows:

``` {yaml}
schedules_path: "examples/data/synthetic_schedules.csv"
```

We are keen to test different encodings (such as continuous sequences versus discretized time-steps).

The encoder and it's parameters are defined in the config `encoder` group. For example:

```{yaml}
encoder_params:
  name: "discrete"
  duration: 1440
  step_size: 10
```

See more examples in `caveat/configs`.

More encoders are defined in the `encoders` module and should be accessed via `caveat.encoders.library`.

Note that encoders must implement both an encode and decode method so that model outputs can be converted back into the population format for reporting.

### Attributes

If a conditional model is being trained, an `attributes_path` should also be specified:

``` {yaml}
attributes_path: "examples/data/synthetic_attributes.csv"
```

By default a conditional model will generate a synthetic model using the input attributes configured above. Alternately an alternative population of attributes can be configured using a `synthetic_attributes_path`:

``` {yaml}
synthetic_attributes_path: "examples/data/some_other_attributes.csv"
```

The encoding of attributes is controlled using the conditionals module, for example:

``` {yaml}
conditionals:
  gender: nominal
  age: {ordinal: [0,100]}
  employment: nominal
```

## Model

The model and it's parameters are defined in the config `model_params` group. For example:

```{yaml}
model_params:
  name: "conv"
  hidden_layers: [64,64]
  latent_dim: 2
  stride: [2,2]
```

See more examples in `caveat.configs`.

Models are defined in `models` and should be accessed via `caveat.models.library`. Models and their training should be specified via the config.

The `data_loader`, `experiment` and `trainer` hyper-params are also configured by similarly named groups. These groups use the standard [pytorch-lightning](https://pypi.org/project/pytorch-lightning/) framework.

## Evaluate

### Basic Evaluation

Each model (with weights from the best performing validation step) is used to generate a new "synthetic" population of schedules. These "synthetic" populations are evaluated by comparing them to an "target" population of schedules. By default the input schedules are used as this target. Alternately a new schedules path can be configures using the evaluation params:

```{yaml}
evaluation_params:
    schedules_path: "examples/data/synthetic_schedules.csv"
```

Evaluating the quality of generated populations is subjective. The `features` module provides functions for extracting features from populations. Such as "activity durations". These are then used to make descriptive metrics and distance metrics between the observed and synthetic populations.

See [examples](https://github.com/fredshone/caveat/latest/examples) for additional evaluation inspiration.

### Conditional Evaluation

When evaluating conditionality, ie the distributions between sequences and attributes, additional configuarion is required to segment sequences based on attributes. Evaluation is then made for each segmentation (or *sub-population*). For example, to evaluate based on different gender and employment attributes:

```{yaml}
evaluation_params:
    schedules_path: "examples/data/synthetic_schedules.csv"  # this will default to the input schedules
    split_on: [gender, employment]
```

Note that the gender and employment sub-populations are not joint, ie there is **not** splitting by gender **and** employment simulataneously.

<!--- --8<-- [end:docs] -->

## Documentation

For more detailed instructions, see our [documentation](https://fredshone.github.io/caveat/latest).

## Installation

To install caveat, we recommend using the [mamba](https://mamba.readthedocs.io/en/latest/index.html) package manager:

<!--- --8<-- [start:docs-install-dev] -->
``` shell
git clone git@github.com:fredshone/caveat.git
cd caveat
mamba create -n caveat -c conda-forge -c city-modelling-lab -c pytorch --file requirements/base.txt --file requirements/dev.txt
mamba activate caveat
pip install --no-deps -e .
```

Caveat is in development, hence an "editable" (`-e`) install is recommended.

### Jupyter Notebooks

To run the example notebooks you will need to add a ipython kernel into the mamba environemnt: `ipython kernel install --user --name=caveat`.

### Windoes and CUDA
If you want to get a cuda enabled windows install you can try the following mamba create:
```
mamba create -n caveat -c conda-forge -c city-modelling-lab -c pytorch -c nvidia --file requirements/cuda_base.txt --file requirements/dev.txt
```
Or lake a look [here](https://pytorch.org/get-started/locally/).
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