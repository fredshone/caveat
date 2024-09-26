import datetime
from pathlib import Path

from torch.random import seed as seeder

from caveat.run import (
    encode_input_attributes,
    encode_schedules,
    evaluate_synthetics,
    generate,
    initiate_logger,
    load_data,
    run_test,
    test_inference,
    train,
)


def jrun_command(
    config: dict,
    verbose: bool = False,
    gen: bool = True,
    test: bool = False,
    infer=True,
) -> None:
    """
    Runs the training for joint-model variation.

    Args:
        config (dict): A dictionary containing the configuration parameters.

    Returns:
        None
    """
    attribute_encoder = config.get("attribute_encoder", None)
    if attribute_encoder is None or attribute_encoder != "tokens":
        raise ValueError(
            "Joint model requires attribute_encoder to be configured as 'tokens'."
        )

    conditionals = config.get("conditionals", None)
    if conditionals is None:
        raise ValueError("No conditionals provided in the config.")

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
        label_encoder=attribute_encoder,
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

    if infer:
        test_infer_path = Path(f"{logger.log_dir}/test_inference")
        test_infer_path.mkdir(exist_ok=True, parents=True)

        test_inference(
            trainer=trainer,
            schedule_encoder=schedule_encoder,
            attribute_encoder=attribute_encoder,
            write_dir=test_infer_path,
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
        synthetic_schedules, _, _ = generate(
            trainer=trainer,
            population=synthetic_population,
            schedule_encoder=schedule_encoder,
            attribute_encoder=attribute_encoder,
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
