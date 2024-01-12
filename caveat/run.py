import datetime
from pathlib import Path
from typing import Union

import torch
from pandas import DataFrame
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger
from torch.random import seed as seeder

from caveat import cuda_available, data, encoders, models, report
from caveat.experiment import Experiment


def run_command(config: dict) -> None:
    """
    Runs the training and reporting process using the provided configuration.

    Args:
        config (dict): A dictionary containing the configuration parameters.

    Returns:
        None
    """
    logger_params = config.get("logging_params", {})
    log_dir = Path(logger_params.get("log_dir", "logs"))
    name = str(
        logger_params.get(
            "name", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )
    )
    write_path = log_dir / name

    data_path = Path(config["data_path"])
    observed = data.load_and_validate(data_path)

    print(f"Loaded {len(observed)} sequences from {data_path}")

    logger = initiate_logger(log_dir, name)
    seed = config.pop("seed", seeder())
    trainer, data_encoder = train(name, observed, config, logger, seed)
    sampled = {
        name: sample(
            trainer, len(observed), data_encoder, config, logger.log_dir, seed
        )
    }

    report.report(observed, sampled, write_path)


def batch_command(batch_config: dict, stats: bool = False) -> None:
    """
    Runs a batch of training and reporting runs based on the provided configuration.

    Args:
        batch_config (dict[dict]): A dictionary containing the configuration for each training job.

    Returns:
        None
    """
    global_config = batch_config.pop("global")
    logger_params = global_config.get("logging_params", {})
    name = str(
        logger_params.get(
            "name", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )
    )
    log_dir = Path(logger_params.get("log_dir", "logs"), name)

    data_path = global_config["data_path"]
    observed = data.load_and_validate(data_path)
    print(f"Loaded {len(observed)} sequences from {data_path}")

    sampled = {}
    for name, config in batch_config.items():
        name = str(name)
        logger = initiate_logger(log_dir, name)
        combined_config = global_config.copy()
        combined_config.update(config)
        write_path = log_dir / name
        seed = combined_config.pop("seed", seeder())
        trainer, encoder = train(name, observed, combined_config, logger, seed)
        sampled[name] = sample(
            trainer, len(observed), encoder, config, write_path, seed
        )

    report.report(observed, sampled, log_dir, report_stats=stats)


def nrun_command(config: dict, n: int = 5, stats: bool = False) -> None:
    """
    Repeat a single model training while varying the seed.

    Args:
        config (dict): A dictionary containing the configuration parameters.
        n (int, optional): The number of times to repeat the run. Defaults to 5.
    """
    logger_params = config.get("logging_params", {})

    name = str(
        logger_params.get(
            "name", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )
    )
    log_dir = Path(logger_params.get("log_dir", "logs")) / name

    data_path = Path(config["data_path"])
    observed = data.load_and_validate(data_path)

    sampled = {}
    for i in range(n):
        run_name = f"{name}_nrun{i}"
        logger = initiate_logger(log_dir, run_name)
        seed = seeder()
        trainer, encoder = train(run_name, observed, config, logger, seed)
        sampled[name] = sample(
            trainer, len(observed), encoder, config, log_dir / run_name, seed
        )

    report.report(observed, sampled, log_dir, report_stats=stats)


def nsample_command(config: dict, n: int = 5, stats: bool = False) -> None:
    """
    Repeat a single run with multiple samples varying the seed.

    Args:
        config (dict): A dictionary containing the configuration parameters.
        n (int, optional): The number of times to repeat the run. Defaults to 5.
    """
    logger_params = config.get("logging_params", {})
    name = str(
        logger_params.get(
            "name", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )
    )
    log_dir = Path(logger_params.get("log_dir", "logs"))
    logger = initiate_logger(log_dir, name)

    data_path = Path(config["data_path"])
    observed = data.load_and_validate(data_path)

    seed = config.pop("seed", seeder())
    trainer, encoder = train(name, observed, config, logger, seed)

    sampled = {}
    for i in range(n):
        run_name = f"{name}_nsample{i}"
        seed = seeder()
        sampled[name] = sample(
            trainer,
            len(observed),
            encoder,
            config,
            log_dir / name / run_name,
            seed,
        )

    report.report(observed, sampled, log_dir, report_stats=stats)


def report_command(
    observed_path: Path,
    log_dir: Path,
    name: str = "synthetic.csv",
    verbose: bool = False,
    head: int = 10,
    batch: bool = False,
):
    observed_path = Path(observed_path)
    log_dir = Path(log_dir)
    observed = data.load_and_validate(observed_path)
    sampled = {}
    if batch:
        paths = [p for p in log_dir.iterdir() if p.is_dir()]
    else:
        paths = [log_dir]

    for experiment in paths:
        # get most recent version
        version = sorted([d for d in experiment.iterdir() if d.is_dir()])[-1]
        path = experiment / version.name / name
        sampled[experiment.name] = data.load_and_validate(path)

    report.report(
        observed=observed,
        sampled=sampled,
        log_dir=log_dir,
        report_description=True,
        report_scores=True,
        report_creativity=True,
        verbose=verbose,
        head=head,
    )


def train(
    name: str,
    observed: DataFrame,
    config: dict,
    logger: TensorBoardLogger,
    seed: int = None,
) -> DataFrame:
    """
    Trains a model on the observed data. Return model trainer (which includes model) and encoder.

    Args:
        name (str): The name of the experiment.
        observed (pandas.DataFrame): The "observed" population data to train the model on.
        config (dict): A dictionary containing the configuration parameters for the experiment.
        logger (TensorBoardLogger): Logger.

    Returns:
        Tuple(pytorch.Trainer, BaseEncoder).
    """
    torch.manual_seed(seed)

    if cuda_available():
        torch.set_float32_matmul_precision("medium")

    print(f"\n======= Training {name} =======")

    observed_sample = data.sample_observed(observed, config)
    data_encoder = build_encoder(config)
    encoded = data_encoder.encode(observed_sample)
    data_loader = build_dataloader(config, encoded)

    experiment = build_experiment(encoded, config)
    trainer = build_trainer(logger, config)
    trainer.fit(experiment, datamodule=data_loader)

    return trainer, data_encoder


def sample(
    trainer: Trainer,
    population_size: int,
    data_encoder: encoders.BaseEncoder,
    config: dict,
    write_dir: Path,
    seed: int,
) -> DataFrame:
    torch.manual_seed(seed)
    print("\n======= Sampling =======")
    predict_loader = data.predict_dataloader(
        population_size, config["model_params"]["latent_dim"], 256
    )
    predictions = trainer.predict(ckpt_path="best", dataloaders=predict_loader)
    predictions = torch.concat(predictions)  # type: ignore
    synthetic = data_encoder.decode(encoded=predictions)
    data.validate(synthetic)
    synthesis_path = write_dir / "synthetic.csv"
    synthetic.to_csv(synthesis_path)
    return synthetic


def build_encoder(config: dict) -> encoders.BaseEncoder:
    encoder_name = config["encoder_params"]["name"]
    data_encoder = encoders.library[encoder_name](**config["encoder_params"])
    return data_encoder


def build_dataloader(
    config: dict, dataset: encoders.BaseEncodedPlans
) -> data.DataModule:
    data_loader_params = config.get("loader_params", {})
    datamodule = data.DataModule(data=dataset, **data_loader_params)
    datamodule.setup()
    return datamodule


def build_experiment(
    dataset: encoders.BaseEncodedPlans, config: dict
) -> Experiment:
    model_name = config["model_params"]["name"]
    model = models.library[model_name]
    model = model(
        in_shape=dataset.shape(),
        encodings=dataset.encodings,
        encoding_weights=dataset.encoding_weights,
        **config["model_params"],
    )
    return Experiment(model, **config["experiment_params"])


def build_trainer(logger: TensorBoardLogger, config: dict) -> Trainer:
    trainer_config = config.get("trainer_params", {})
    patience = trainer_config.pop("patience", 5)
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(logger.log_dir, "checkpoints"),
        monitor="val_recon_loss",
        save_top_k=2,
        save_weights_only=True,
    )
    return Trainer(
        logger=logger,
        callbacks=[
            EarlyStopping(
                monitor="val_recon_loss",
                patience=patience,
                stopping_threshold=0.0,
            ),
            LearningRateMonitor(),
            checkpoint_callback,
        ],
        **trainer_config,
    )


def initiate_logger(save_dir: Union[Path, str], name: str) -> TensorBoardLogger:
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
