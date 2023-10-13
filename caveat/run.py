from pathlib import Path

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# from pytorch_lightning.plugins import DDPPlugin
from caveat import models
from caveat.data.loader import VAEDataset
from caveat.experiment import Experiment

tb_logger = TensorBoardLogger(save_dir="logs", name="testVAE")
Path(f"{tb_logger.log_dir}/samples").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.log_dir}/reconstructions").mkdir(exist_ok=True, parents=True)

model = models.VAE(
    in_shape=(1, 5, 144), latent_dim=2, hidden_dims=[64, 64], stride=(2, 2)
)
experiment = Experiment(model)
# summary = ModelSummary(experiment)

data = VAEDataset(path="~/Projects/caveat/examples/example_population.csv")
data.setup()
print(data.mapping)

runner = Trainer(
    logger=tb_logger,
    callbacks=[
        LearningRateMonitor(),
        ModelCheckpoint(
            save_top_k=2,
            dirpath=Path(tb_logger.log_dir, "checkpoints"),
            monitor="val_loss",
            save_last=True,
        ),
    ],
    max_epochs=20,
)

print("======= Training =======")
runner.fit(experiment, datamodule=data)
