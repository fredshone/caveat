from pathlib import Path

import pytest

from caveat.data import load_and_validate


@pytest.fixture
def observed():
    data_path = Path("tests/fixtures/synthetic_population.csv")
    return load_and_validate(data_path)


@pytest.fixture
def run_config_one_hot():
    return {
        "logging_params": {"log_dir": "tmp", "name": "test"},
        "encoder_params": {
            "name": "one_hot",
            "step_size": 10,
            "duration": 1440,
        },
        "loader_params": {
            "train_batch_size": 2,
            "val_batch_size": 2,
            "num_workers": 2,
        },
        "experiment_params": {
            "LR": 0.005,
            "weight_decay": 0.0,
            "scheduler_gamma": 0.95,
            "kld_weight": 0.025,
        },
        "trainer_params": {"max_epochs": 1, "min_epochs": 1},
        "seed": 1234,
        "model_params": {
            "name": "Conv2d",
            "hidden_layers": [1],
            "latent_dim": 2,
            "dropout": 0.1,
        },
    }


@pytest.fixture
def run_config_embed_cov():
    return {
        "logging_params": {"log_dir": "tmp", "name": "test"},
        "encoder_params": {
            "name": "descrete",
            "step_size": 10,
            "duration": 1440,
        },
        "loader_params": {
            "train_batch_size": 2,
            "val_batch_size": 2,
            "num_workers": 2,
        },
        "experiment_params": {
            "LR": 0.005,
            "weight_decay": 0.0,
            "scheduler_gamma": 0.95,
            "kld_weight": 0.025,
        },
        "trainer_params": {"max_epochs": 1, "min_epochs": 1},
        "seed": 1234,
        "model_params": {
            "name": "EmbedConv",
            "hidden_layers": [1],
            "latent_dim": 2,
            "dropout": 0.1,
        },
    }


@pytest.fixture
def run_config_gru():
    return {
        "logging_params": {"log_dir": "tmp", "name": "test"},
        "encoder_params": {
            "name": "sequence",
            "max_length": 4,
            "norm_duration": 1440,
        },
        "loader_params": {
            "train_batch_size": 2,
            "val_batch_size": 2,
            "num_workers": 2,
        },
        "experiment_params": {
            "LR": 0.005,
            "weight_decay": 0.0,
            "scheduler_gamma": 0.95,
            "kld_weight": 0.025,
            "duration_weight": 10,
        },
        "trainer_params": {"max_epochs": 1, "min_epochs": 1},
        "seed": 1234,
        "model_params": {
            "name": "GRU",
            "hidden_layers": 1,
            "hidden_size": 1,
            "latent_dim": 2,
            "teacher_forcing_ratio": 0.5,
            "use_mask": True,
            "dropout": 0.1,
        },
    }


@pytest.fixture
def run_config_lstm():
    return {
        "logging_params": {"log_dir": "tmp", "name": "test"},
        "encoder_params": {
            "name": "sequence",
            "max_length": 4,
            "norm_duration": 1440,
        },
        "loader_params": {
            "train_batch_size": 2,
            "val_batch_size": 2,
            "num_workers": 2,
        },
        "experiment_params": {
            "LR": 0.005,
            "weight_decay": 0.0,
            "scheduler_gamma": 0.95,
            "kld_weight": 0.025,
            "duration_weight": 10,
        },
        "trainer_params": {"max_epochs": 1, "min_epochs": 1},
        "seed": 1234,
        "model_params": {
            "name": "LSTM",
            "hidden_layers": 1,
            "hidden_size": 1,
            "latent_dim": 2,
            "teacher_forcing_ratio": 0.5,
            "use_mask": True,
            "dropout": 0.1,
        },
    }


@pytest.fixture
def batch_config():
    return {
        "global": {
            "data_path": "tests/fixtures/synthetic_population.csv",
            "logging_params": {"log_dir": "tmp", "name": "test"},
            "encoder_params": {
                "name": "sequence",
                "max_length": 4,
                "norm_duration": 1440,
            },
            "loader_params": {
                "train_batch_size": 2,
                "val_batch_size": 2,
                "num_workers": 2,
            },
            "experiment_params": {
                "LR": 0.005,
                "weight_decay": 0.0,
                "scheduler_gamma": 0.95,
                "kld_weight": 0.025,
                "duration_weight": 10,
            },
            "trainer_params": {"max_epochs": 2, "min_epochs": 2},
            "seed": 1234,
        },
        "conv": {
            "model_params": {
                "name": "Conv",
                "hidden_layers": 1,
                "hidden_size": 1,
                "latent_dim": 2,
                "teacher_forcing_ratio": 0.5,
                "use_mask": True,
                "dropout": 0.1,
            }
        },
        "gru": {
            "model_params": {
                "name": "GRU",
                "hidden_layers": 1,
                "hidden_size": 1,
                "latent_dim": 2,
                "teacher_forcing_ratio": 0.5,
                "use_mask": True,
                "dropout": 0.1,
            }
        },
        "lstm": {
            "model_params": {
                "name": "LSTM",
                "hidden_layers": 1,
                "hidden_size": 1,
                "latent_dim": 2,
                "teacher_forcing_ratio": 0.5,
                "use_mask": True,
                "dropout": 0.1,
            }
        },
    }
