<!--- the "--8<--" html comments define what part of the README to add to the index page of the documentation -->
<!--- --8<-- [start:docs] -->
![caveat](resources/logos/title.png)

Deep learning for modelling human activity schedules.

[![Daily CI Build](https://github.com/fredshone/caveat/actions/workflows/daily-scheduled-ci.yml/badge.svg)](https://github.com/fredshone/caveat/actions/workflows/daily-scheduled-ci.yml)
[![Documentation](https://github.com/fredshone/caveat/actions/workflows/pages/pages-build-deployment/badge.svg)](https://fredshone.github.io/caveat)

Caveat is for building and evaluating models of human activity schedules. Activity schedules are a useful representation of human behaviours, used for modelling transport, energy and epidemiological systems.

Activity scheduling is required component of activity-based models. But Caveat also has application for (i) diverse upsampling, (ii) bias correction, (iii) simple forecasting.

## Research

Caveat is part of an ongoing research project. Overview of presented work can be found in the [papers](https://github.com/fredshone/caveat/blob/main/papers/README.md) module, including key information allowing presented results to be **reproduced**.

## Example Applications

**Generation** for up-sampling or anonymisation an observed population of activity schedules:
- `caveat run configs/example_run.yaml` - train a generative model using an observed population of schedules, sample a new synthetic population of schedules, and evaluate it's quality.

**Conditional Generation** for bias correction and simple forecasting from an observed population of schedules and attributes:
- `caveat run configs/example_run_conditional.yaml` - train a model using an observed population of schedules with attributes, sample a new population conditional on new person attributes, and evaluate it's quality.

**Predict**
- `caveat run configs/example_seq2score.yaml --test` - train and test a model to predict MATSim plan scores.
- `caveat run configs/example_seq2seq.yaml --test` - train and test a model to predict an output MATSim plan from input.

## Quick Start

Once [installed](#installation) get started using `caveat --help`.

Caveat provides various commands to facilitate rapid and reproducible experiments. Outputs are written to the (default) `logs` directory:

### Run

`caveat run --help`

Train and evaluate a model. The run data, encoder, model and other parameters are controlled using a run config. For example:

- `caveat run configs/example_run.yaml`
- `caveat run configs/example_run_conditional.yaml` (conditional generation)

### Batch

`caveat batch --help`

Batch allows training and comparison of multiple models and/or hyper-params as per a batch config. For example:

- `caveat batch configs/example_batch.yaml`
- `caveat batch configs/example_batch_conditional.yaml` (conditional generation)

### Nrun

`caveat nrun --help`

Nrun is a simplified version of batch used to repeat the same model training and evaluation. This is intended to test for variance in model training and sampling. For example, run and evaluate the variance of _n=3_ of the same experiment using:

- `caveat nrun configs/example_run.yaml --n 3`

The config is as per a regular run config but `seed` is ignored.

### Ngen

`caveat ngen --help`

As per nrun but only assesses variance from the sampling/generative process (not model training):

- `caveat ngen configs/example_run.yaml --n 3`

### Reporting

`caveat report --help`

Evaluate the outputs of an existing run (or batch using `-b`).

### Logging

Caveat writes tensorboard logs to a (default) `logs/` directory. Monitor or review training progress using `tensorboard --logdir=logs`.

## Configuration

For help with configuration refer to our [documentation](https://fredshone.github.io/caveat/latest/configuration).

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

See our [documentation](https://fredshone.github.io/caveat/latest/contributing/).

## Building the documentation

If you are unable to access the online documentation, you can build the documentation locally.
First, [install a development environment of caveat](https://fredshone.github.io/caveat/latest/installation/), then deploy the documentation using [mike](https://github.com/jimporter/mike):

```
mike deploy develop
mike serve
```

Then you can view the documentation in a browser at http://localhost:8000/.


## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [arup-group/cookiecutter-pypackage](https://github.com/arup-group/cookiecutter-pypackage) project template.