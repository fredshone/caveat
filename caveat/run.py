from pathlib import Path

import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger
from torch import manual_seed

from caveat import data, encoders, models, report
from caveat.experiment import Experiment


def runner(config: dict):
    print("======= Data Setup =======")
    data_path = config["data_params"].pop("data_path")
    encoder_name = config["data_params"].pop("encoding")

    observed = data.load_and_validate(data_path)
    data_encoder = encoders.library[encoder_name](**config["data_params"])
    encoded = data_encoder.encode(observed)

    data_loader_params = config.get("loader_params", {})
    datamodule = encoders.DataModule(data=encoded, **data_loader_params)
    datamodule.setup()

    print("======= Model Setup =======")
    manual_seed(config.get("seed", 1234))

    model_name = config["model_params"]["model"]
    model = models.library[model_name]

    model = model(in_shape=encoded.shape(), **config["model_params"])
    experiment = Experiment(model, **config["experiment_params"])

    logger = initiate_logger(config)

    runner = Trainer(
        logger=logger,
        callbacks=[
            EarlyStopping(
                monitor="val_reconstruction_loss",
                patience=3,
                stopping_threshold=0.0,
            ),
            LearningRateMonitor(),
            ModelCheckpoint(
                save_top_k=2,
                dirpath=Path(logger.log_dir, "checkpoints"),
                monitor="val_loss",
                save_last=True,
            ),
        ],
        strategy="ddp",
        **config.get("trainer_params", {}),
    )

    print("======= Training =======")
    runner.fit(experiment, datamodule=datamodule)

    print("======= Evaluating =======")
    best = Experiment.load_from_checkpoint(
        Path(logger.log_dir, "checkpoints", "last.ckpt"),
        model=model,
        **config["experiment_params"],
    )
    y = best.sample(len(encoded))
    y = data_encoder.decode(encoded=y)
    data.validate(y)
    synthesis_path = Path(experiment.logger.log_dir, "synthetic.csv")
    y.to_csv(synthesis_path)

    df = pd.concat(
        [
            report.check_activity_start_and_ends(observed, y),
            report.activity_participation_rates(observed, y),
            report.av_activity_durations(observed, y),
            report.av_activity_start_times(observed, y),
            report.av_activity_end_times(observed, y),
        ]
    )
    df.style.format(
        {
            "observed": "{:.2f}",
            "synth": "{:.2f}",
            "delta": "{:.2f}",
            "perc": "{:.2%}",
        }
    )

    print(df.to_markdown())


def initiate_logger(config: dict) -> TensorBoardLogger:
    model_name = config["model_params"]["model"]
    logging_params = config.get("logging_params", {})
    save_dir = logging_params.get("save_dir", "logs")
    log_name = logging_params.get("name", model_name)
    tb_logger = TensorBoardLogger(save_dir=save_dir, name=log_name)
    Path(f"{tb_logger.log_dir}/samples").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/reconstructions").mkdir(
        exist_ok=True, parents=True
    )
    return tb_logger
