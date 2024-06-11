import datetime
import pickle
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
from torch import Tensor
from torch.random import seed as seeder

from caveat import cuda_available, data, encoding, models
from caveat.data.module import DataModule
from caveat.encoding import BaseDataset, BaseEncoder
from caveat.evaluate import evaluate
from caveat.experiment import Experiment


def run_command(
    config: dict, verbose: bool = False, gen: bool = True, test: bool = False
) -> None:
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
    logger = initiate_logger(log_dir, name)
    seed = config.pop("seed", seeder())

    # load data
    input_schedules, input_attributes, synthetic_attributes = load_data(config)

    # encode data
    attribute_encoder, encoded_attributes = encode_input_attributes(
        logger.log_dir, input_attributes, config
    )

    schedule_encoder, encoded_schedules, data_loader = encode_schedules(
        logger.log_dir, input_schedules, encoded_attributes, config
    )

    # train
    trainer = train(
        name=name,
        data_loader=data_loader,
        encoded_schedules=encoded_schedules,
        config=config,
        test=test,
        gen=gen,
        logger=logger,
        seed=seed,
    )

    if test:
        # test the model
        run_test(
            trainer=trainer,
            schedule_encoder=schedule_encoder,
            write_dir=Path(logger.log_dir),
            seed=seed,
        )

    if gen:
        # prepare synthetic attributes
        if synthetic_attributes is not None:
            synthetic_population = attribute_encoder.encode(
                synthetic_attributes
            )
        else:
            synthetic_population = input_schedules.pid.nunique()

        # generate synthetic schedules
        synthetic_schedules = generate(
            trainer=trainer,
            population=synthetic_population,
            schedule_encoder=schedule_encoder,
            config=config,
            write_dir=Path(logger.log_dir),
            seed=seed,
        )

        # evaluate synthetic schedules
        evaluate_synthetics(
            synthetic_schedules={name: synthetic_schedules},
            synthetic_attributes={name: synthetic_attributes},
            default_eval_schedules=input_schedules,
            default_eval_attributes=input_attributes,
            write_path=Path(logger.log_dir),
            eval_params=config.get("evaluation_params", {}),
            stats=False,
            verbose=verbose,
        )


def batch_command(
    batch_config: dict,
    stats: bool = True,
    verbose: bool = False,
    test: bool = False,
    gen: bool = True,
) -> None:
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

    synthetic_schedules = {}
    synthetic_attributes_all = {}

    for name, config in batch_config.items():
        name = str(name)
        logger = initiate_logger(log_dir, name)

        # build config for this run
        combined_config = global_config.copy()
        combined_config.update(config)
        seed = combined_config.pop("seed", seeder())

        # load data
        input_schedules, input_attributes, synthetic_attributes = load_data(
            combined_config
        )

        # encode data
        attribute_encoder, encoded_attributes = encode_input_attributes(
            logger.log_dir, input_attributes, combined_config
        )

        schedule_encoder, encoded_schedules, data_loader = encode_schedules(
            logger.log_dir, input_schedules, encoded_attributes, combined_config
        )

        # train
        trainer = train(
            name=name,
            data_loader=data_loader,
            encoded_schedules=encoded_schedules,
            config=combined_config,
            test=test,
            gen=gen,
            logger=logger,
            seed=seed,
        )
        if test:
            # test the model
            run_test(
                trainer=trainer,
                schedule_encoder=schedule_encoder,
                write_dir=Path(logger.log_dir),
                seed=seed,
            )
        if gen:
            # prepare synthetic attributes
            if synthetic_attributes is not None:
                synthetic_population = attribute_encoder.encode(
                    synthetic_attributes
                )
            else:
                synthetic_population = input_schedules.pid.nunique()

            # record synthetic attributes for evaluation
            synthetic_attributes_all[name] = synthetic_attributes

            # generate synthetic schedules
            synthetic_schedules[name] = generate(
                trainer=trainer,
                population=synthetic_population,
                schedule_encoder=schedule_encoder,
                config=combined_config,
                write_dir=Path(logger.log_dir),
                seed=seed,
            )
    if gen:
        # evaluate synthetic schedules
        evaluate_synthetics(
            synthetic_schedules=synthetic_schedules,
            synthetic_attributes=synthetic_attributes_all,
            default_eval_schedules=input_schedules,
            default_eval_attributes=input_attributes,
            write_path=logger.log_dir,
            eval_params=global_config.get("evaluation_params", {}),
            stats=stats,
            verbose=verbose,
        )


def nrun_command(
    config: dict,
    n: int = 5,
    stats: bool = True,
    verbose: bool = False,
    test: bool = False,
    gen: bool = True,
) -> None:
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

    # load data
    input_schedules, input_attributes, synthetic_attributes = load_data(config)

    # encode data
    attribute_encoder, encoded_attributes = encode_input_attributes(
        log_dir, input_attributes, config
    )

    schedule_encoder, encoded_schedules, data_loader = encode_schedules(
        log_dir, input_schedules, encoded_attributes, config
    )

    synthetic_schedules = {}
    all_synthetic_attributes = {}

    for i in range(n):
        run_name = f"{name}_nrun{i}"
        logger = initiate_logger(log_dir, run_name)
        seed = seeder()
        trainer = train(
            name=run_name,
            data_loader=data_loader,
            encoded_schedules=encoded_schedules,
            config=config,
            test=test,
            gen=gen,
            logger=logger,
            seed=seed,
        )
        if test:
            run_test(
                trainer=trainer,
                schedule_encoder=schedule_encoder,
                write_dir=Path(logger.log_dir),
                seed=seed,
            )
        if gen:
            # prepare synthetic attributes
            if synthetic_attributes is not None:
                synthetic_population = attribute_encoder.encode(
                    synthetic_attributes
                )
            else:
                synthetic_population = input_schedules.pid.nunique()

            all_synthetic_attributes[run_name] = synthetic_attributes

            synthetic_schedules[run_name] = generate(
                trainer=trainer,
                population=synthetic_population,
                schedule_encoder=schedule_encoder,
                config=config,
                write_dir=Path(logger.log_dir),
                seed=seed,
            )

    if gen:
        evaluate_synthetics(
            synthetic_schedules=synthetic_schedules,
            synthetic_attributes=all_synthetic_attributes,
            default_eval_schedules=input_schedules,
            default_eval_attributes=input_attributes,
            write_path=log_dir,
            eval_params=config.get("evaluation_params", {}),
            stats=stats,
            verbose=verbose,
        )


def ngen_command(
    config: dict, n: int = 5, stats: bool = False, verbose: bool = False
) -> None:
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
    training_logger = initiate_logger(log_dir, name)

    # load data
    input_schedules, input_attributes, synthetic_attributes = load_data(config)

    # encode data
    attribute_encoder, encoded_attributes = encode_input_attributes(
        log_dir, input_attributes, config
    )

    schedule_encoder, encoded_schedules, data_loader = encode_schedules(
        log_dir, input_schedules, encoded_attributes, config
    )

    seed = config.pop("seed", seeder())

    # train
    trainer = train(
        name=name,
        data_loader=data_loader,
        encoded_schedules=encoded_schedules,
        config=config,
        test=False,
        gen=True,
        logger=training_logger,
        seed=seed,
    )

    synthetic_schedules = {}
    all_synthetic_attributes = {}

    # prepare synthetic attributes
    if synthetic_attributes is not None:
        synthetic_population = attribute_encoder.encode(synthetic_attributes)
    else:
        synthetic_population = input_schedules.pid.nunique()

    for i in range(n):
        logger = initiate_logger(training_logger.log_dir, f"nsample{i}")
        seed = seeder()
        synthetic_schedules[f"nsample{i}"] = generate(
            trainer=trainer,
            population=synthetic_population,
            schedule_encoder=schedule_encoder,
            config=config,
            write_dir=Path(logger.log_dir),
            seed=seed,
        )
        all_synthetic_attributes[f"nsample{i}"] = synthetic_attributes

    evaluate_synthetics(
        synthetic_schedules=synthetic_schedules,
        synthetic_attributes=all_synthetic_attributes,
        default_eval_schedules=input_schedules,
        default_eval_attributes=input_attributes,
        write_path=log_dir,
        eval_params=config.get("evaluation_params", {}),
        stats=stats,
        verbose=verbose,
    )


def report_command(
    observed_path: Path,
    log_dir: Path,
    name: str = "synthetic.csv",
    verbose: bool = False,
    head: int = 10,
    batch: bool = False,
    stats: bool = True,
):
    observed_path = Path(observed_path)
    log_dir = Path(log_dir)
    observed = data.load_and_validate_schedules(observed_path)
    synthetic_schedules = {}
    if batch:
        paths = [p for p in log_dir.iterdir() if p.is_dir()]
    else:
        paths = [log_dir]

    for experiment in paths:
        # get most recent version
        version = sorted([d for d in experiment.iterdir() if d.is_dir()])[-1]
        path = experiment / version.name / name
        synthetic_schedules[experiment.name] = data.load_and_validate_schedules(
            path
        )

    reports = evaluate.evaluate(
        target_schedules=observed,
        synthetic_schedules=synthetic_schedules,
        report_stats=stats,
    )
    evaluate.report(reports, log_dir=log_dir, head=head, verbose=verbose)


def load_data(config: dict) -> Tuple[DataFrame, DataFrame, DataFrame]:
    # load schedules data
    schedules_path = Path(config["schedules_path"])
    schedules = data.load_and_validate_schedules(schedules_path)
    print(f"Loaded {schedules.pid.nunique()} schedules from {schedules_path}")

    # load attributes data (conditional case)
    attributes, synthetic_attributes = data.load_and_validate_attributes(
        config, schedules
    )
    if attributes is not None:
        print(
            f"Loaded {len(attributes)} attributes from {config['attributes_path']}"
        )
    return schedules, attributes, synthetic_attributes


def encode_schedules(
    log_dir: Path,
    schedules: DataFrame,
    attributes: Optional[Tensor],
    config: dict,
) -> Tuple[BaseEncoder, BaseDataset, DataModule]:

    # encode schedules
    schedule_encoder = build_encoder(config)
    encoded_schedules = schedule_encoder.encode(
        schedules=schedules, conditionals=attributes
    )
    data_loader = build_dataloader(config, encoded_schedules)

    pickle.dump(schedule_encoder, open(f"{log_dir}/schedule_encoder.pkl", "wb"))

    return (schedule_encoder, encoded_schedules, data_loader)


def encode_input_attributes(
    log_dir: Path, input_attributes: Optional[DataFrame], config: dict
) -> Tuple[BaseEncoder, BaseDataset, DataModule, Tensor]:
    attribute_encoder = None
    # optionally encode attributes
    if input_attributes is not None:
        conditionals_config = config.get("conditionals", None)
        if conditionals_config is None:
            raise UserWarning("Config must contain conditionals configuration.")
        attribute_encoder = encoding.AttributeEncoder(conditionals_config)
        input_attributes = attribute_encoder.encode(input_attributes)

    pickle.dump(
        attribute_encoder, open(f"{log_dir}/attribute_encoder.pkl", "wb")
    )

    return (attribute_encoder, input_attributes)


def train(
    name: str,
    data_loader: DataModule,
    encoded_schedules: BaseDataset,
    config: dict,
    test: bool,
    gen: bool,
    logger: TensorBoardLogger,
    seed: Optional[int] = None,
) -> Tuple[Trainer, encoding.BaseEncoder]:
    """
    Trains a model on the observed data. Return model trainer (which includes model) and encoder.

    Args:
        name (str): The name of the experiment.
        schedules (pandas.DataFrame): The "observed" population data to train the model on.
        conditionals (pandas.DataFrame): The "conditionals" data to train the model on.
        config (dict): A dictionary containing the configuration parameters for the experiment.
        logger (TensorBoardLogger): Logger.

    Returns:
        Tuple(pytorch.Trainer, BaseEncoder).
    """
    print(f"\n======= Training {name} =======")

    torch.manual_seed(seed)

    if cuda_available():
        torch.set_float32_matmul_precision("medium")

    torch.cuda.empty_cache()
    experiment = build_experiment(encoded_schedules, config, test, gen)
    trainer = build_trainer(logger, config)
    trainer.fit(experiment, datamodule=data_loader)
    return trainer


def run_test(
    trainer: Trainer,
    schedule_encoder: encoding.BaseEncoder,
    write_dir: Path,
    seed: int,
    ckpt_path: Optional[str] = None,
):
    torch.manual_seed(seed)
    print("\n======= Testing =======")
    if ckpt_path is None:
        ckpt_path = "best"
    trainer.test(ckpt_path=ckpt_path, datamodule=trainer.datamodule)
    (test_in, test_target, conditionals, predictions) = zip(
        *list(
            trainer.predict(
                ckpt_path="best",
                dataloaders=trainer.datamodule.test_dataloader(),
            )
        )
    )
    test_in = torch.concat(test_in)
    test_target = torch.concat(test_target)
    conditionals = torch.concat(conditionals)
    predictions = torch.concat(predictions)

    test_in = schedule_encoder.decode_input(schedules=test_in)
    data.validate_schedules(test_in)
    test_in.to_csv(write_dir / "test_input.csv")

    test_target = schedule_encoder.decode_target(schedules=test_target)
    test_target.to_csv(write_dir / "test_target.csv")

    predictions = schedule_encoder.decode_output(schedules=predictions)
    predictions.to_csv(write_dir / "pred.csv")

    return test_in, test_target, predictions


def generate(
    trainer: Trainer,
    population: Union[int, Tensor],
    schedule_encoder: encoding.BaseEncoder,
    config: dict,
    write_dir: Path,
    seed: int,
    ckpt_path: Optional[str] = None,
) -> DataFrame:
    torch.manual_seed(seed)
    if ckpt_path is None:
        ckpt_path = "best"
    latent_dims = config.get("model_params", {}).get(
        "latent_dim", 2
    )  # default of 2
    batch_size = config.get("generator_params", {}).get("batch_size", 256)
    if isinstance(population, int):
        print(f"\n======= Sampling {population} new schedules =======")
        predictions = generate_n(
            trainer,
            n=population,
            batch_size=batch_size,
            latent_dims=latent_dims,
            seed=seed,
            ckpt_path=ckpt_path,
        )
    elif isinstance(population, Tensor):
        print(
            f"\n======= Sampling {len(population)} new schedules from synthetic attributes ======="
        )
        predictions = generate_from_attributes(
            trainer,
            attributes=population,
            batch_size=batch_size,
            latent_dims=latent_dims,
            seed=seed,
            ckpt_path=ckpt_path,
        )

    synthetic = schedule_encoder.decode(schedules=predictions)
    data.validate_schedules(synthetic)
    synthesis_path = write_dir / "synthetic.csv"
    synthetic.to_csv(synthesis_path)
    return synthetic


def generate_n(
    trainer: Trainer,
    n: int,
    batch_size: int,
    latent_dims: int,
    seed: int,
    ckpt_path: str,
) -> torch.Tensor:
    torch.manual_seed(seed)
    dataloaders = data.build_predict_dataloader(n, latent_dims, batch_size)
    predictions = trainer.predict(ckpt_path=ckpt_path, dataloaders=dataloaders)
    predictions = torch.concat(predictions)
    return predictions


def generate_from_attributes(
    trainer: Trainer,
    attributes: Tensor,
    batch_size: int,
    latent_dims: int,
    seed: int,
    ckpt_path: str,
) -> torch.Tensor:
    torch.manual_seed(seed)
    dataloaders = data.build_conditional_dataloader(
        attributes, latent_dims, batch_size
    )
    predictions = trainer.predict(ckpt_path=ckpt_path, dataloaders=dataloaders)
    predictions = torch.concat(predictions)
    return predictions


def evaluate_synthetics(
    synthetic_schedules: dict[str, DataFrame],
    synthetic_attributes: dict[str, DataFrame],
    default_eval_schedules: DataFrame,
    default_eval_attributes: DataFrame,
    write_path: Path,
    eval_params: dict,
    stats: bool = True,
    verbose: bool = False,
) -> None:
    head = eval_params.get("head", 10)

    eval_schedules_path = eval_params.get("schedules_path", None)
    if eval_schedules_path:
        eval_schedules = data.load_and_validate_schedules(eval_schedules_path)
        print(
            f"Loaded {len(eval_schedules)} schedules for evaluation from {eval_schedules_path}"
        )
    else:
        eval_schedules = default_eval_schedules
        print("Evaluating synthetic schedules against input schedules")

    split_on = eval_params.get("split_on", [])
    if split_on:
        eval_attributes_path = eval_params.get("attributes_path", None)
        if eval_attributes_path:
            eval_attributes = data.load_and_validate_attributes(
                eval_params, eval_schedules
            )
            print(
                f"Loaded {len(eval_attributes)} attributes for evaluation from {eval_attributes_path}"
            )
        else:
            eval_attributes = default_eval_attributes

        sub_reports = evaluate.evaluate_subsampled(
            synthetic_schedules=synthetic_schedules,
            synthetic_attributes=synthetic_attributes,
            target_schedules=eval_schedules,
            target_attributes=eval_attributes,
            split_on=split_on,
            report_stats=stats,
        )
        evaluate.report_splits(
            sub_reports,
            log_dir=write_path,
            head=head,
            verbose=verbose,
            suffix="_subs",
        )

    else:
        reports = evaluate.evaluate(
            target_schedules=eval_schedules,
            synthetic_schedules=synthetic_schedules,
            report_stats=stats,
        )
        evaluate.report(reports, log_dir=write_path, head=head, verbose=verbose)


def conditional_sample(
    trainer: Trainer,
    population_size: int,
    data_encoder: encoding.BaseEncoder,
    config: dict,
    write_dir: Path,
    seed: int,
) -> DataFrame:
    torch.manual_seed(seed)
    print("\n======= Sampling =======")
    predict_loader = data.build_predict_dataloader(
        population_size, config["model_params"]["latent_dim"], 256
    )
    predictions = trainer.predict(ckpt_path="best", dataloaders=predict_loader)
    predictions = torch.concat(predictions)  # type: ignore
    synthetic = data_encoder.decode(schedules=predictions)
    data.validate_schedules(synthetic)
    synthesis_path = write_dir / "synthetic.csv"
    synthetic.to_csv(synthesis_path)
    return synthetic


def build_encoder(config: dict) -> encoding.BaseEncoder:
    encoder_name = config["encoder_params"]["name"]
    data_encoder = encoding.library[encoder_name](**config["encoder_params"])
    return data_encoder


def build_dataloader(
    config: dict, dataset: encoding.BaseDataset
) -> data.DataModule:
    data_loader_params = config.get("loader_params", {})
    datamodule = data.DataModule(data=dataset, **data_loader_params)
    datamodule.setup()
    return datamodule


def build_experiment(
    dataset: encoding.BaseDataset, config: dict, test: bool, gen: bool
) -> Experiment:
    model_name = config["model_params"]["name"]
    model = models.library[model_name]
    model = model(
        in_shape=dataset.shape(),
        encodings=dataset.activity_encodings,
        encoding_weights=dataset.encoding_weights,
        conditionals_size=dataset.conditionals_shape,
        **config["model_params"],
    )
    return Experiment(
        model, test=test, gen=gen, **config.get("experiment_params", {})
    )


def build_trainer(logger: TensorBoardLogger, config: dict) -> Trainer:
    trainer_config = config.get("trainer_params", {})
    patience = trainer_config.pop("patience", 5)
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(logger.log_dir, "checkpoints"),
        monitor="val_loss",
        save_top_k=2,
        save_weights_only=True,
    )
    return Trainer(
        logger=logger,
        callbacks=[
            EarlyStopping(
                monitor="val_loss", patience=patience, stopping_threshold=0.0
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
