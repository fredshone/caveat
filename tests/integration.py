from caveat.run import (
    build_dataloader,
    build_encoder,
    build_experiment,
    build_trainer,
    initiate_logger,
)


def test_descrete_one_hot(tmp_path, observed, run_config_one_hot):
    logger = initiate_logger(tmp_path, "test")

    data_encoder = build_encoder(run_config_one_hot)
    encoded = data_encoder.encode(observed)
    data_loader = build_dataloader(run_config_one_hot, encoded)

    experiment = build_experiment(encoded, run_config_one_hot)
    trainer = build_trainer(logger, run_config_one_hot)

    trainer.validate(model=experiment, dataloaders=data_loader)


def test_descrete_embed_conv(tmp_path, observed, run_config_embed_cov):
    logger = initiate_logger(tmp_path, "test")

    data_encoder = build_encoder(run_config_embed_cov)
    encoded = data_encoder.encode(observed)
    data_loader = build_dataloader(run_config_embed_cov, encoded)

    experiment = build_experiment(encoded, run_config_embed_cov)
    trainer = build_trainer(logger, run_config_embed_cov)

    trainer.validate(model=experiment, dataloaders=data_loader)


def test_gru(tmp_path, observed, run_config_gru):
    logger = initiate_logger(tmp_path, "test")

    data_encoder = build_encoder(run_config_gru)
    encoded = data_encoder.encode(observed)
    data_loader = build_dataloader(run_config_gru, encoded)

    experiment = build_experiment(encoded, run_config_gru)
    trainer = build_trainer(logger, run_config_gru)

    trainer.validate(model=experiment, dataloaders=data_loader)


def test_lstm(tmp_path, observed, run_config_lstm):
    logger = initiate_logger(tmp_path, "test")

    data_encoder = build_encoder(run_config_lstm)
    encoded = data_encoder.encode(observed)
    data_loader = build_dataloader(run_config_lstm, encoded)

    experiment = build_experiment(encoded, run_config_lstm)
    trainer = build_trainer(logger, run_config_lstm)

    trainer.validate(model=experiment, dataloaders=data_loader)
