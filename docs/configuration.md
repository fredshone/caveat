# Configuration

To get started we recommend using an existing example config, such as `example_run.yaml` or `example_batch.yaml` for a batch command.

## Input Data

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

- **pid** (person id) field is a unique identifier designating the attributes for each above sequence

Other than the `pid` column, columns can be arbitrarily named and represented with any data type.

### Examples

There are example [toy populations](https://github.com/fredshone/caveat/latest/examples/data) with 1000 sequences in `caveat/examples/data`. There are also [example notebooks](https://github.com/fredshone/caveat/tree/main/examples) for:

- Generation of a synthetic population
- Generation of a population from UK travel diaries (requires access to UK NTS trip data)

## Modules

Caveat provides a framework to train and evaluate models. Caveat is composed of the following modules:

- [Encoding](#encoding) - encoding/decoding inputs into various data representations
- [Model](#model) - generative models to learn and synthesize new populations of sequences

For generative models you may also want to use [Evaluate](#evaluate) to measuring the quality of synthetic populations of sequences.

Caveat is designed to be installed and run as a command line application. The above modules are configured using config files. Such that experiments are accessible without code and can be reproduced.

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

## Experiment Hyper-parameters

The `data_loader`, `experiment` and `trainer` hyper-params are also configured by similarly named groups. These groups use the standard [pytorch-lightning](https://pypi.org/project/pytorch-lightning/) framework.

Please refer to the example config files or ask if in doubt.

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

When evaluating conditionality, ie the distributions between sequences and attributes, additional configuration is required to segment sequences based on attributes. Evaluation is then made for each segmentation (or *sub-population*). For example, to evaluate based on different gender and employment attributes:

```{yaml}
evaluation_params:
    schedules_path: "examples/data/synthetic_schedules.csv"  # this will default to the input schedules
    split_on: [gender, employment]
```

Note that the gender and employment sub-populations are not joint, ie there is **not** splitting by gender **and** employment simultaneously.
