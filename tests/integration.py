from caveat.run import batch_command, nrun_command, nsample_command, run_command


def test_run_conv(run_config_embed_cov):
    run_command(run_config_embed_cov)


def test_run_lstm(run_config_lstm):
    run_command(run_config_lstm)


def test_run_cond_lstm(run_config_conditional_lstm):
    run_command(run_config_conditional_lstm)


def test_batch_multi_model(batch_config):
    batch_command(batch_config)


def test_nrun(run_config_lstm):
    nrun_command(run_config_lstm)


def test_nsample(run_config_lstm):
    nsample_command(run_config_lstm)
