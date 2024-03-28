from caveat.run import (
    build_attributes,
    build_dataloader,
    build_encoder,
    build_experiment,
    build_trainer,
    initiate_logger,
)


def test_descrete_embed_conv(tmp_path, test_schedules, run_config_embed_cov):
    logger = initiate_logger(tmp_path, "test")

    data_encoder = build_encoder(run_config_embed_cov)
    encoded = data_encoder.encode(test_schedules, conditionals=None)
    data_loader = build_dataloader(run_config_embed_cov, encoded)

    experiment = build_experiment(encoded, run_config_embed_cov)
    trainer = build_trainer(logger, run_config_embed_cov)

    trainer.validate(model=experiment, dataloaders=data_loader)


def test_gru(tmp_path, test_schedules, run_config_gru):
    logger = initiate_logger(tmp_path, "test")

    data_encoder = build_encoder(run_config_gru)
    encoded = data_encoder.encode(test_schedules, conditionals=None)
    data_loader = build_dataloader(run_config_gru, encoded)

    experiment = build_experiment(encoded, run_config_gru)
    trainer = build_trainer(logger, run_config_gru)

    trainer.validate(model=experiment, dataloaders=data_loader)


def test_lstm(tmp_path, test_schedules, run_config_lstm):
    logger = initiate_logger(tmp_path, "test")

    data_encoder = build_encoder(run_config_lstm)
    encoded = data_encoder.encode(test_schedules, conditionals=None)
    data_loader = build_dataloader(run_config_lstm, encoded)

    experiment = build_experiment(encoded, run_config_lstm)
    trainer = build_trainer(logger, run_config_lstm)

    trainer.validate(model=experiment, dataloaders=data_loader)


def test_conditional_lstm(
    tmp_path, test_schedules, test_attributes, run_config_conditional_lstm
):
    logger = initiate_logger(tmp_path, "test")

    schedule_encoder = build_encoder(run_config_conditional_lstm)
    _, attributes, _ = build_attributes(
        run_config_conditional_lstm, test_schedules
    )
    encoded = schedule_encoder.encode(test_schedules, conditionals=attributes)
    data_loader = build_dataloader(run_config_conditional_lstm, encoded)

    experiment = build_experiment(encoded, run_config_conditional_lstm)
    trainer = build_trainer(logger, run_config_conditional_lstm)

    trainer.validate(model=experiment, dataloaders=data_loader)
