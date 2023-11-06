import datetime
from pathlib import Path

from pandas import DataFrame, concat, set_option
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger
from torch import manual_seed

from caveat import data, encoders, features, models, report
from caveat.experiment import Experiment


def run(config: dict):
    logger_params = config.get("logging_params", {})
    log_dir = Path(logger_params.get("log_dir", "logs"))
    name = logger_params.get(
        "name", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )

    seed = config.pop("seed", 1234)
    manual_seed(seed)

    data_path = Path(config["data_path"])
    observed = data.load_and_validate(data_path)

    sampled = {
        name: train_sample_and_report(name, observed, config, log_dir, seed)
    }

    report_results(observed, sampled, log_dir)


def batch(batch_config: dict[dict]):
    global_config = batch_config.pop("global")
    logger_params = global_config.get("logging_params", {})
    name = logger_params.get(
        "name", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    log_dir = Path(logger_params.get("log_dir", "logs"), name)

    seed = global_config.pop("seed", 1234)
    manual_seed(seed)

    data_path = global_config["data_path"]
    observed = data.load_and_validate(data_path)

    sampled = {
        name: train_sample_and_report(name, observed, config, log_dir, seed)
        for name, config in batch_config.items()
    }

    report_results(observed, sampled, log_dir)


def train_sample_and_report(
    name: str, observed: DataFrame, config: dict, log_dir: Path, seed: int
):
    print(f"======= Training {name.title()} =======")
    logger = initiate_logger(log_dir, name)

    encoder_name = config["encoder_params"].pop("encoding")
    data_encoder = encoders.library[encoder_name](**config["encoder_params"])
    encoded = data_encoder.encode(observed)

    data_loader_params = config.get("loader_params", {})
    datamodule = data.DataModule(data=encoded, **data_loader_params)
    datamodule.setup()

    model_name = config["model_params"]["model"]
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
        strategy="ddp",
        **config.get("trainer_params", {}),
    )

    runner.fit(experiment, datamodule=datamodule)

    print(f"======= Sampling {name.title()} =======")
    best_path = checkpoint_callback.best_model_path
    print(f"Loading best model from {best_path}")
    best = Experiment.load_from_checkpoint(
        best_path, model=model, **config["experiment_params"]
    )
    synthetic = best.generate(len(encoded))
    synthetic = data_encoder.decode(encoded=synthetic)
    data.validate(synthetic)
    synthesis_path = Path(experiment.logger.log_dir, "synthetic.csv")
    synthetic.to_csv(synthesis_path)
    return synthetic


def report_results(
    observed: DataFrame, sampled: dict[str, DataFrame], log_dir: Path
):
    df = concat(
        [
            report.report_diff(
                observed, sampled, features.structural.start_and_end_acts
            ),
            report.report_diff(
                observed, sampled, features.participation.participation_rates
            ),
            report.report_diff(
                observed,
                sampled,
                features.participation.act_plan_seq_participation_rates,
            ).head(10),
            report.report_diff(
                observed,
                sampled,
                features.participation.act_seq_participation_rates,
            ).head(10),
            report.report_diff(
                observed,
                sampled,
                features.participation.joint_participation_rates,
            ).head(10),
            report.report_diff(
                observed, sampled, features.transitions.transition_rates
            ),
            report.report_diff(
                observed, sampled, features.sequence.sequence_probs
            ),
            report.report_diff(
                observed, sampled, features.durations.average_activity_durations
            ),
            report.report_diff(
                observed,
                sampled,
                features.durations.average_activity_plan_seq_durations,
            ).head(10),
            report.report_diff(
                observed,
                sampled,
                features.durations.average_activity_seq_durations,
            ).head(10),
            report.report_diff(
                observed, sampled, features.times.average_start_times
            ),
            report.report_diff(
                observed, sampled, features.times.average_end_times
            ),
        ]
    )
    set_option("display.precision", 2)
    df.to_csv(Path(log_dir, "report.csv"))
    print(df.to_markdown())


def initiate_logger(save_dir, name) -> TensorBoardLogger:
    tb_logger = TensorBoardLogger(save_dir=save_dir, name=name)
    Path(f"{tb_logger.log_dir}/samples").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/reconstructions").mkdir(
        exist_ok=True, parents=True
    )
    return tb_logger
