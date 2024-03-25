import datetime
from pathlib import Path
from typing import Optional, Tuple, Union

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

from caveat import cuda_available, data, encoders, evaluate, models
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

    # load schedules data
    data_path = Path(config["data_path"])
    schedules = data.load_and_validate_schedules(data_path)
    print(f"Loaded {len(schedules)} sequences from {data_path}")

    # load attributes data (conditional case)
    attributes, synthetic_attributes = data.load_and_validate_attributes(
        config, schedules
    )

    logger = initiate_logger(log_dir, name)
    seed = config.pop("seed", seeder())
    trainer, data_encoder = train(
        name,
        observed=schedules,
        conditionals=attributes,
        config=config,
        logger=logger,
        seed=seed,
    )
    sampled = {
        name: generate(
            trainer,
            len(schedules),
            data_encoder,
            config,
            Path(logger.log_dir),
            seed,
        )
    }

    evaluate.report(schedules, sampled, write_path)


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
    observed = data.load_and_validate_schedules(data_path)
    print(f"Loaded {len(observed)} sequences from {data_path}")

    sampled = {}
    for name, config in batch_config.items():
        name = str(name)
        logger = initiate_logger(log_dir, name)
        combined_config = global_config.copy()
        combined_config.update(config)
        seed = combined_config.pop("seed", seeder())
        trainer, encoder = train(name, observed, combined_config, logger, seed)
        sampled[name] = generate(
            trainer,
            observed.pid.nunique(),
            encoder,
            combined_config,
            Path(logger.log_dir),
            seed,
        )

    evaluate.report(observed, sampled, log_dir, report_stats=stats)


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
    observed = data.load_and_validate_schedules(data_path)

    sampled = {}
    for i in range(n):
        run_name = f"{name}_nrun{i}"
        logger = initiate_logger(log_dir, run_name)
        seed = seeder()
        trainer, encoder = train(run_name, observed, config, logger, seed)
        sampled[name] = generate(
            trainer,
            observed.pid.nunique(),
            encoder,
            config,
            Path(logger.log_dir),
            seed,
        )

    evaluate.report(observed, sampled, log_dir, report_stats=stats)


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
    observed = data.load_and_validate_schedules(data_path)

    seed = config.pop("seed", seeder())
    trainer, encoder = train(name, observed, config, logger, seed)

    sampled = {}
    for i in range(n):
        sample_dir = Path(logger.log_dir) / f"nsample{i}"
        seed = seeder()
        sampled[name] = generate(
            trainer, observed.pid.nunique(), encoder, config, sample_dir, seed
        )

    evaluate.report(observed, sampled, log_dir, report_stats=stats)


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
    observed = data.load_and_validate_schedules(observed_path)
    sampled = {}
    if batch:
        paths = [p for p in log_dir.iterdir() if p.is_dir()]
    else:
        paths = [log_dir]

    for experiment in paths:
        # get most recent version
        version = sorted([d for d in experiment.iterdir() if d.is_dir()])[-1]
        path = experiment / version.name / name
        sampled[experiment.name] = data.load_and_validate_schedules(path)

    evaluate.report(
        observed=observed,
        sampled=sampled,
        log_dir=log_dir,
        verbose=verbose,
        head=head,
    )


def train(
    name: str,
    observed: DataFrame,
    conditionals: Optional[DataFrame],
    config: dict,
    logger: TensorBoardLogger,
    seed: Optional[int] = None,
) -> Tuple[Trainer, encoders.BaseEncoder]:
    """
    Trains a model on the observed data. Return model trainer (which includes model) and encoder.

    Args:
        name (str): The name of the experiment.
        observed (pandas.DataFrame): The "observed" population data to train the model on.
        conditionals (pandas.DataFrame): The "conditionals" data to train the model on.
        config (dict): A dictionary containing the configuration parameters for the experiment.
        logger (TensorBoardLogger): Logger.

    Returns:
        Tuple(pytorch.Trainer, BaseEncoder).
    """
    torch.manual_seed(seed)

    if cuda_available():
        torch.set_float32_matmul_precision("medium")

    torch.cuda.empty_cache()

    print(f"\n======= Training {name} =======")

    observed_schedules, observed_conditionals = data.sample_data(
        observed, conditionals, config
    )

    data_encoder = build_encoder(config)
    encoded = data_encoder.encode(
        schedules=observed_schedules, conditionals=observed_conditionals
    )
    data_loader = build_dataloader(config, encoded)

    experiment = build_experiment(encoded, config)
    trainer = build_trainer(logger, config)
    trainer.fit(experiment, datamodule=data_loader)

    return trainer, data_encoder


def generate(
    trainer: Trainer,
    population: Union[int, DataFrame],
    data_encoder: encoders.BaseEncoder,
    config: dict,
    write_dir: Path,
    seed: int,
) -> DataFrame:
    torch.manual_seed(seed)
    latent_dims = config["model_params"]["latent_dim"]
    batch_size = config.get("generator_params", {}).get("batch_size", 256)
    if isinstance(population, int):
        print(f"\n======= Sampling {population} new schedules =======")
        predictions = generate_n(
            trainer,
            n=population,
            batch_size=batch_size,
            latent_dims=latent_dims,
            seed=seed,
        )
    elif isinstance(population, DataFrame):
        print(
            f"\n======= Sampling {len(population)} new schedules from attributes ======="
        )
        predictions = generate_from_attributes(
            trainer,
            attributes=population,
            batch_size=batch_size,
            latent_dims=latent_dims,
            seed=seed,
        )

    synthetic = data_encoder.decode(schedules=predictions)
    data.validate_schedules(synthetic)
    synthesis_path = write_dir / "synthetic.csv"
    synthetic.to_csv(synthesis_path)
    return synthetic


def generate_n(
    trainer: Trainer, n: int, batch_size: int, latent_dims: int, seed: int
) -> torch.Tensor:
    torch.manual_seed(seed)
    predict_loader = data.predict_dataloader(n, latent_dims, batch_size)
    predictions = trainer.predict(ckpt_path="best", dataloaders=predict_loader)
    predictions = torch.concat(predictions)  # type: ignore
    return predictions


def generate_from_attributes(
    trainer: Trainer,
    attributes: DataFrame,
    batch_size: int,
    latent_dims: int,
    seed: int,
) -> torch.Tensor:
    torch.manual_seed(seed)
    predict_loader = data.predict_dataloader(
        attributes, latent_dims, batch_size
    )
    predictions = trainer.predict(ckpt_path="best", dataloaders=predict_loader)
    predictions = torch.concat(predictions)  # type: ignore
    return predictions


def conditional_sample(
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
    synthetic = data_encoder.decode(schedules=predictions)
    data.validate_schedules(synthetic)
    synthesis_path = write_dir / "synthetic.csv"
    synthetic.to_csv(synthesis_path)
    return synthetic


def build_encoder(config: dict) -> encoders.BaseEncoder:
    encoder_name = config["encoder_params"]["name"]
    data_encoder = encoders.library[encoder_name](**config["encoder_params"])
    return data_encoder


def build_dataloader(
    config: dict, dataset: encoders.BaseEncoded
) -> data.DataModule:
    data_loader_params = config.get("loader_params", {})
    datamodule = data.DataModule(data=dataset, **data_loader_params)
    datamodule.setup()
    return datamodule


def build_experiment(dataset: encoders.BaseEncoded, config: dict) -> Experiment:
    model_name = config["model_params"]["name"]
    model = models.library[model_name]
    model = model(
        in_shape=dataset.shape(),
        encodings=dataset.encodings,
        encoding_weights=dataset.encoding_weights,
        conditionals_size=dataset.conditionals_shape,
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
