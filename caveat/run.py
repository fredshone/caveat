import datetime
from pathlib import Path
from typing import Dict, Optional

import torch
from pandas import DataFrame, concat, set_option
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger
from torch.random import seed as seeder

from caveat import cuda_available, data, encoders, features, models, report
from caveat.experiment import Experiment


def run(config: Dict) -> None:
    """
    Runs the training and reporting process using the provided configuration.

    Args:
        config (dict): A dictionary containing the configuration parameters.

    Returns:
        None
    """
    logger_params = config.get("logging_params", {})
    log_dir = Path(logger_params.get("log_dir", "logs"))
    name = logger_params.get(
        "name", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    write_path = log_dir / name

    seed = config.pop("seed", seeder())

    data_path = Path(config["data_path"])
    observed = data.load_and_validate(data_path)

    sampled = {
        name: train_sample_and_report(name, observed, config, log_dir, seed)
    }

    report_results(observed, sampled, write_path)


def batch(batch_config: dict[dict]):
    """
    Runs a batch of training and reporting runs based on the provided configuration.

    Args:
        batch_config (dict[dict]): A dictionary containing the configuration for each training job.

    Returns:
        None
    """
    global_config = batch_config.pop("global")
    logger_params = global_config.get("logging_params", {})
    name = logger_params.get(
        "name", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    log_dir = Path(logger_params.get("log_dir", "logs"), name)

    seed = batch_config.pop("seed", seeder())

    data_path = global_config["data_path"]
    observed = data.load_and_validate(data_path)

    sampled = {
        name: train_sample_and_report(name, observed, config, log_dir, seed)
        for name, config in batch_config.items()
    }

    report_results(observed, sampled, log_dir)


def nrun(config: Dict, n: int = 5):
    """
    Repeat a single run while varying the seed, report on variance.

    Args:
        config (dict): A dictionary containing the configuration parameters.
        n (int, optional): The number of times to repeat the run. Defaults to 5.
    """
    logger_params = config.get("logging_params", {})
    log_dir = Path(logger_params.get("log_dir", "logs"))
    name = logger_params.get(
        "name", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    write_path = log_dir / name

    data_path = Path(config["data_path"])
    observed = data.load_and_validate(data_path)

    sampled = {
        f"{name}_{i}": train_sample_and_report(
            f"{name}_{i}", observed, config, log_dir
        )
        for i in range(n)
    }

    describe_results(observed, sampled, write_path)


def train_sample_and_report(
    name: str,
    observed: DataFrame,
    config: dict,
    log_dir: Path,
    seed: Optional[int] = None,
):
    """
    Trains a model on the observed data, generates synthetic data using the trained model,
    and saves the synthetic data. Returns the synthetic data as a population DataFrame.

    Args:
        name (str): The name of the experiment.
        observed (pandas.DataFrame): The "observed" population data to train the model on.
        config (dict): A dictionary containing the configuration parameters for the experiment.
        log_dir (pathlib.Path): The directory to save the experiment logs and checkpoints.
        seed (int, optional): The random seed to use for the experiment. Defaults to None.

    Returns:
        pandas.DataFrame: The synthetic data generated by the trained model.
    """

    if cuda_available():
        torch.set_float32_matmul_precision("medium")
    if seed is None:
        seed = seeder()
    torch.manual_seed(seed)
    print(f"======= Training {name.title()} =======")
    logger = initiate_logger(log_dir, name)

    encoder_name = config["encoder_params"]["name"]
    data_encoder = encoders.library[encoder_name](**config["encoder_params"])
    encoded = data_encoder.encode(observed)

    data_loader_params = config.get("loader_params", {})
    datamodule = data.DataModule(data=encoded, **data_loader_params)
    datamodule.setup()

    model_name = config["model_params"]["name"]
    model = models.library[model_name]
    model = model(in_shape=encoded.shape(), **config["model_params"])
    experiment = Experiment(model, **config["experiment_params"])

    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(logger.log_dir, "checkpoints"),
        monitor="val_loss",
        save_top_k=1,
        save_weights_only=True,
    )

    runner = Trainer(
        logger=logger,
        callbacks=[
            EarlyStopping(
                monitor="val_reconstruction_loss",
                patience=3,
                stopping_threshold=0.0,
            ),
            LearningRateMonitor(),
            checkpoint_callback,
        ],
        **config.get("trainer_params", {}),
    )

    runner.fit(experiment, datamodule=datamodule)

    print(f"======= Sampling {name.title()} =======")
    predict_loader = data.predict_dataloader(
        len(encoded), config["model_params"]["latent_dim"], 256
    )
    predictions = runner.predict(ckpt_path="best", dataloaders=predict_loader)
    predictions = torch.concat(predictions)
    synthetic = data_encoder.decode(encoded=predictions)
    data.validate(synthetic)
    synthesis_path = Path(experiment.logger.log_dir, "synthetic.csv")
    synthetic.to_csv(synthesis_path)
    return synthetic


def report_results(
    observed: DataFrame,
    sampled: dict[str, DataFrame],
    log_dir: Path,
    n: int = 10,
):
    """
    Generate a report of the differences between the observed and sampled dataframes.

    Args:
        observed (DataFrame): The observed population.
        sampled (dict[str, DataFrame]): A dictionary of sampled populations.
        log_dir (Path): The directory to save the report CSV file.
        n (int, optional): The number of rows to include in the report for each feature. Defaults to 10.
    """
    df = concat(
        [
            report.report_diff(
                observed, sampled, features.structural.start_and_end_acts
            ).head(n),
            report.report_diff(
                observed, sampled, features.participation.participation_rates
            ).head(n),
            report.report_diff(
                observed,
                sampled,
                features.participation.act_plan_seq_participation_rates,
            ).head(n),
            report.report_diff(
                observed,
                sampled,
                features.participation.act_seq_participation_rates,
            ).head(n),
            report.report_diff(
                observed,
                sampled,
                features.participation.joint_participation_rates,
            ).head(n),
            report.report_diff(
                observed, sampled, features.transitions.transition_rates
            ).head(n),
            report.report_diff(
                observed, sampled, features.sequence.sequence_probs
            ).head(n),
            report.report_diff(
                observed, sampled, features.durations.average_activity_durations
            ).head(n),
            report.report_diff(
                observed,
                sampled,
                features.durations.average_activity_plan_seq_durations,
            ).head(n),
            report.report_diff(
                observed,
                sampled,
                features.durations.average_activity_seq_durations,
            ).head(n),
            report.report_diff(
                observed, sampled, features.times.average_start_times
            ).head(n),
            report.report_diff(
                observed, sampled, features.times.average_end_times
            ).head(n),
        ]
    )
    set_option("display.precision", 2)
    df.to_csv(Path(log_dir, "report.csv"))
    print(df.to_markdown())


def describe_results(
    observed: DataFrame,
    sampled: dict[str, DataFrame],
    log_dir: Path,
    n: int = 10,
):
    """
    Generate a variance report of the observed and sampled data.

    Parameters:
    observed (DataFrame): The observed population.
    sampled (dict[str, DataFrame]): A dictionary of synthetic populations, where the keys are the names of the models.
    log_dir (Path): The directory where the report CSV file will be saved.
    n (int): The number of rows to include in each section of the report. Default is 10.

    Returns:
    None
    """
    df = concat(
        [
            report.describe(
                observed, sampled, features.structural.start_and_end_acts
            ).head(n),
            report.describe(
                observed, sampled, features.participation.participation_rates
            ).head(n),
            report.describe(
                observed,
                sampled,
                features.participation.act_plan_seq_participation_rates,
            ).head(n),
            report.describe(
                observed,
                sampled,
                features.participation.act_seq_participation_rates,
            ).head(n),
            report.describe(
                observed,
                sampled,
                features.participation.joint_participation_rates,
            ).head(n),
            report.describe(
                observed, sampled, features.transitions.transition_rates
            ).head(n),
            report.describe(
                observed, sampled, features.sequence.sequence_probs
            ).head(n),
            report.describe(
                observed, sampled, features.durations.average_activity_durations
            ).head(n),
            report.describe(
                observed,
                sampled,
                features.durations.average_activity_plan_seq_durations,
            ).head(n),
            report.describe(
                observed,
                sampled,
                features.durations.average_activity_seq_durations,
            ).head(n),
            report.describe(
                observed, sampled, features.times.average_start_times
            ).head(n),
            report.describe(
                observed, sampled, features.times.average_end_times
            ).head(n),
        ]
    )
    set_option("display.precision", 2)
    write_path = log_dir / "describe.csv"
    print(write_path)
    df.to_csv(write_path)
    print(df.to_markdown())


def initiate_logger(save_dir: str, name: str) -> TensorBoardLogger:
    """
    Initializes a TensorBoardLogger object for logging training progress.

    Args:
        save_dir (str): The directory where the logs will be saved.
        name (str): The name of the logger.

    Returns:
        TensorBoardLogger: The initialized TensorBoardLogger object.
    """
    tb_logger = TensorBoardLogger(save_dir=save_dir, name=name)
    Path(f"{tb_logger.log_dir}/samples").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/reconstructions").mkdir(
        exist_ok=True, parents=True
    )
    return tb_logger