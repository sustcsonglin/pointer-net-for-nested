# wandb
from pytorch_lightning.loggers import WandbLogger
import wandb

# pytorch
from pytorch_lightning import Callback
import pytorch_lightning as pl
import torch

# others
import glob
import os


def get_wandb_logger(trainer: pl.Trainer) -> WandbLogger:
    logger = None
    for lg in trainer.logger:
        if isinstance(lg, WandbLogger):
            logger = lg

    if not logger:
        raise Exception(
            "You are using wandb related callback,"
            "but WandbLogger was not found for some reason..."
        )

    return logger


# class UploadCodeToWandbAsArtifact(Callback):
#     """Upload all *.py files to wandb as an artifact at the beginning of the run."""
#
#     def __init__(self, code_dir: str):
#         self.code_dir = code_dir
#
#     def on_train_start(self, trainer, pl_module):
#         logger = get_wandb_logger(trainer=trainer)
#         experiment = logger.experiment
#
#         code = wandb.Artifact("project-source", type="code")
#         for path in glob.glob(os.path.join(self.code_dir, "**/*.py"), recursive=True):
#             print(path)
#             code.add_file(path)
#             print('ok')
#
#
#
#         experiment.use_artifact(code)
#         print("successfully update the code .")


class UploadHydraConfigFileToWandb(Callback):
    def on_fit_start(self, trainer, pl_module: LightningModule) -> None:
        logger = get_wandb_logger(trainer=trainer)

        logger.experiment.save()



class UploadCheckpointsToWandbAsArtifact(Callback):
    """Upload experiment checkpoints to wandb as an artifact at the end of training."""

    def __init__(self, ckpt_dir: str = "checkpoints/", upload_best_only: bool = False):
        self.ckpt_dir = ckpt_dir
        self.upload_best_only = upload_best_only

    def on_train_end(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        ckpts = wandb.Artifact("experiment-ckpts", type="checkpoints")

        if self.upload_best_only:
            ckpts.add_file(trainer.checkpoint_callback.best_model_path)
        else:
            for path in glob.glob(
                os.path.join(self.ckpt_dir, "**/*.ckpt"), recursive=True
            ):
                ckpts.add_file(path)

        experiment.use_artifact(ckpts)


class WatchModelWithWandb(Callback):
    """Make WandbLogger watch model at the beginning of the run."""

    def __init__(self, log: str = "gradients", log_freq: int = 100):
        self.log = log
        self.log_freq = log_freq

    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        logger.watch(model=trainer.model, log=self.log, log_freq=self.log_freq)

