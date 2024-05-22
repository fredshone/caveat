from caveat.run import batch_command, ngen_command, nrun_command, run_command


def test_run_conv(run_config_embed_cov):
    run_command(run_config_embed_cov)


def test_run_vae_lstm(config_vae_lstm):
    run_command(config_vae_lstm)


def test_run_c_lstm(config_c_lstm):
    run_command(config_c_lstm)


def test_run_cvae_lstm(config_cvae_lstm):
    run_command(config_cvae_lstm)


def test_batch_multi_model(batch_config):
    batch_command(batch_config)


def test_nrun(config_vae_lstm):
    nrun_command(config_vae_lstm)


def test_nsample(config_vae_lstm):
    ngen_command(config_vae_lstm)
