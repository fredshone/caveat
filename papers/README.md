# Papers

The following document details the configuration of key Caveat experiments used for published or presented results.

Configurations are provided but *may* not be maintained. Please use them for reference only.

## hEART 2024

**12th Symposium of the European Association for Research in Transportation**

We present a Variational Auto-Encoder (VAE) for generating acivity schedules from the UK National Travel Survey (NTS). Paper available [here](https://github.com/fredshone/caveat/tree/main/papers/hEART2024.pdf).

The paper presents results from two models; (i) the "Discrete" model using a discretised representation of time with CNN architecture, and (ii) the "Sequence" model using a continuous representation of time with RNN architecture.

### Contributions

1. We present a "continuous" activity schedule encoding, loosely based on a language models.
2. We present a VAE architecture for sequence generation.
3. We compare two models architectures; CNN and RNN based.
4. We propose a comprehensive evaluation framework that considers both model "correctness" and model "creativity".

### Reproducibility

#### Data

Training (and validation) data is used from the UK NTS (available [here](https://beta.ukdataservice.ac.uk/datacatalogue/studies/study?id=5340)). We use the following pre-processing:

- Daily activity schedules are extracted for 2021 (multiple days are taken from the same individual where possible).
- Only include daily schedules that start and end in a home activity.
- Activity types are simplified to `home`, `work`, `other`, `education`, `visit`, `shop`, `escort`, `medical`.

This preprocessing is available [here](https://github.com/fredshone/caveat/blob/main/examples/3_NTS_population_demo.ipynb).

#### Configuration

The following batch configuration can be used to reproduce the results from the model using `caveat batch CONFIG_PATH`. Please note that the paper actually presents results from five runs of each model - with different seeds to explore model stochastics.

```
global:
  schedules_path: PATH_TO_DATA
  logging_params:
    log_dir: "logs"
    name: "heart"
  seed: 1234
  experiment_params:
    LR: 0.005
    weight_decay: 0.0
    scheduler_gamma: 0.95
    kld_weight: 0.0001
  loader_params:
    train_batch_size: 256
    val_batch_size:  256
    num_workers: 4
  trainer_params:
    min_epochs: 10
    patience: 10

discrete_model:
  encoder_params:
    name: "discrete"
    duration: 1440
    step_size: 10
  model_params:
    name: "VAE_Conv_Discrete"
    hidden_layers: [128,128,128,128]
    latent_dim: 6
    stride: [2,2]
    dropout: 0.1

sequence_model:
  encoder_params:
    name: "sequence"
    max_length: 16
    norm_duration: 1440
  model_params:
    name: "VAE_LSTM"
    hidden_layers: 4
    hidden_size: 128
    latent_dim: 6
    dropout: 0.1
    teacher_forcing_ratio: 0.5
```

## MUM 2024

**MATSim User Meeting 2024**

We present and evaluate two surrogate models trained on intermediate MATSim plans (from iterations 50, 100, 150 and 200) from an existing MATSim scenario of Sheffield. Presentation available [here](https://github.com/fredshone/caveat/tree/main/papers/MUM2024.pdf).

### Contributions

1. We demonstrate a sequence based encoding for MATSim plans that includes activites and trips, activity type, trip mode, durations and distances.
2. We demonstrate a model able to approximate the MATSIm mobsim and scoring function - the **seq2score** model.
3. We demonstarte a model able to approximate the MATSIm loop (mobsim, scoring and innvoation) - the **seq2seq** model.

### Reproducibility

#### Data

We use data from an existing MATSim scenario provided by the [Arup City Modelling Lab](https://medium.com/arupcitymodelling). This may be available on request.

MATSim plan preprocessing is available [here](https://github.com/fredshone/caveat/blob/main/examples/7_matsim_seq2score.ipynb) for the **seq2score** model and [here](https://github.com/fredshone/caveat/blob/main/examples/6_matsim_seq2seq.ipynb) for the **seq2seq** model.

#### Seq2score Configuration

```
logging_params:
  log_dir: "logs"
  name: "seq2score"

schedules_path: SCHEDULES_TO_SCORE_PATH
attributes_path: ATTRIBUTES_PATH

conditionals:
  subpopulation: nominal
  hcounty: nominal
  gender: nominal
  workstatus: nominal
  hasLicence: nominal
  hasCar: nominal
  hasBike: nominal
  age_group: nominal

encoder_params:
  name: "seq2score"
  max_length: 16
  norm_duration: 2880

model_params:
  name: "Seq2Score_LSTM"
  hidden_layers: 2
  hidden_size: 128
  dropout: 0.1

loader_params:
  train_batch_size: 256
  val_batch_size:  256
  num_workers: 8
  val_split: 0.1
  test_split: 0.1

experiment_params:
  LR: 0.01
  weight_decay: 0.0
  scheduler_gamma: 0.95

trainer_params:
  min_epochs: 20
  patience: 20
  max_epochs: 100

seed: 1234
```

#### Seq2seq Configuration

```
logging_params:
  log_dir: "logs"
  name: "seq2seq"

schedules_path: SCHEDULES_TO_SCHEDULES_PATH
attributes_path: ATTRIBUTES_PATH

conditionals:
  subpopulation: nominal
  hcounty: nominal
  gender: nominal
  workstatus: nominal
  hasLicence: nominal
  hasCar: nominal
  hasBike: nominal
  age_group: nominal

encoder_params:
  name: "seq2seq"
  max_length: 32
  norm_duration: 2880

model_params:
  name: "Seq2Seq_LSTM"
  hidden_layers: 4
  hidden_size: 128
  dropout: 0.1
  teacher_forcing_ratio: 0.5

loader_params:
  train_batch_size: 256
  val_batch_size:  256
  num_workers: 8
  val_split: 0.1
  test_split: 0.1

experiment_params:
  LR: 0.005
  weight_decay: 0.0
  scheduler_gamma: 0.95

trainer_params:
  min_epochs: 10
  patience: 10
  max_epochs: 100

seed: 1234
```