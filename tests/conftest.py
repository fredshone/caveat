from pathlib import Path

import pandas as pd
import pytest

from caveat.data import load_and_validate_schedules


@pytest.fixture
def test_schedules():
    data_path = Path("tests/fixtures/test_schedules.csv")
    return load_and_validate_schedules(data_path)


@pytest.fixture
def test_attributes():
    data_path = Path("tests/fixtures/test_attributes.csv")
    return pd.read_csv(data_path)


@pytest.fixture
def schedules():
    data_path = Path("tests/fixtures/schedules.csv")
    return load_and_validate_schedules(data_path)


@pytest.fixture
def run_config_conditional_lstm(tmp_path):
    return {
        "schedules_path": "tests/fixtures/test_schedules.csv",
        "attributes_path": "tests/fixtures/test_attributes.csv",
        "conditionals": {"age": {"ordinal": [0, 100]}, "gender": "nominal"},
        "logging_params": {"log_dir": tmp_path, "name": "test"},
        "encoder_params": {
            "name": "sequence",
            "max_length": 4,
            "norm_duration": 1440,
        },
        "loader_params": {
            "train_batch_size": 12,
            "val_batch_size": 12,
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
            "name": "conditional_LSTM",
            "hidden_layers": 2,
            "hidden_size": 8,
            "latent_dim": 2,
            "teacher_forcing_ratio": 0.5,
            "use_mask": True,
            "dropout": 0.1,
        },
    }


@pytest.fixture
def run_config_embed_cov(tmp_path):
    return {
        "schedules_path": "tests/fixtures/test_schedules.csv",
        "logging_params": {"log_dir": tmp_path, "name": "test"},
        "encoder_params": {
            "name": "discrete",
            "step_size": 60,
            "duration": 1440,
        },
        "loader_params": {
            "train_batch_size": 8,
            "val_batch_size": 8,
            "num_workers": 1,
        },
        "experiment_params": {
            "LR": 0.005,
            "weight_decay": 0.0,
            "scheduler_gamma": 0.95,
            "kld_weight": 0.001,
        },
        "trainer_params": {"max_epochs": 1, "min_epochs": 1},
        "seed": 1234,
        "model_params": {
            "name": "conv",
            "hidden_layers": [64, 64],
            "latent_dim": 2,
            "dropout": 0.1,
        },
    }


@pytest.fixture
def run_config_lstm(tmp_path):
    return {
        "schedules_path": "tests/fixtures/test_schedules.csv",
        "logging_params": {"log_dir": tmp_path, "name": "test"},
        "encoder_params": {
            "name": "sequence",
            "max_length": 12,
            "norm_duration": 1440,
        },
        "loader_params": {
            "train_batch_size": 8,
            "val_batch_size": 8,
            "num_workers": 1,
        },
        "experiment_params": {
            "LR": 0.005,
            "weight_decay": 0.0,
            "scheduler_gamma": 0.95,
            "kld_weight": 0.001,
            "duration_weight": 10,
        },
        "trainer_params": {"max_epochs": 1, "min_epochs": 1},
        "seed": 1234,
        "model_params": {
            "name": "LSTM",
            "hidden_layers": 2,
            "hidden_size": 8,
            "latent_dim": 2,
            "teacher_forcing_ratio": 0.5,
            "use_mask": True,
            "dropout": 0.1,
        },
    }


@pytest.fixture
def batch_config(tmp_path):
    return {
        "global": {
            "schedules_path": "tests/fixtures/test_schedules.csv",
            "logging_params": {"log_dir": tmp_path, "name": "test"},
            "loader_params": {
                "train_batch_size": 8,
                "val_batch_size": 8,
                "num_workers": 1,
            },
            "experiment_params": {
                "LR": 0.005,
                "weight_decay": 0.0,
                "scheduler_gamma": 0.95,
                "kld_weight": 0.001,
                "duration_weight": 10,
            },
            "trainer_params": {"max_epochs": 1, "min_epochs": 1},
            "seed": 1234,
        },
        "conv": {
            "encoder_params": {
                "name": "discrete",
                "step_size": 60,
                "duration": 1440,
            },
            "model_params": {
                "name": "conv",
                "hidden_layers": [64, 64],
                "latent_dim": 2,
                "dropout": 0.1,
            },
        },
        "lstm": {
            "encoder_params": {
                "name": "sequence",
                "max_length": 12,
                "norm_duration": 1440,
            },
            "model_params": {
                "name": "LSTM",
                "hidden_layers": 2,
                "hidden_size": 8,
                "latent_dim": 2,
                "teacher_forcing_ratio": 0.5,
                "use_mask": True,
                "dropout": 0.1,
            },
        },
    }
