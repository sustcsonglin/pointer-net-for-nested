import os
import shutil
from pathlib import Path
from typing import List

import hydra
import omegaconf
import pytorch_lightning as pl
from hydra.core.hydra_config import HydraConfig
import wandb
from pytorch_lightning import LightningModule, LightningDataModule, Callback, Trainer
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning import seed_everything

# hydra imports
from omegaconf import DictConfig
from hydra.utils import log
import hydra
from omegaconf import OmegaConf

# normal imports
from typing import List
import warnings
import logging
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import sys
# sys.setdefaultencoding() does not exist, here!

from hydra.experimental import compose



def evaluate(config):

    os.chdir(config.load_from_checkpoint)
    original_overrides = OmegaConf.load(os.path.join(config.load_from_checkpoint, ".hydra/overrides.yaml"))
    current_overrides = HydraConfig.get().overrides.task
    hydra_config = OmegaConf.load(os.path.join(config.load_from_checkpoint, ".hydra/hydra.yaml"))
    # getting the config name from the previous job.
    config_name = hydra_config.hydra.job.config_name
    # concatenating the original overrides with the current overrides
    overrides = original_overrides + current_overrides
    # compose a new config from scratch
    config = compose(config_name, overrides=overrides)
    print(config)
    checkpoint = os.path.join(config.load_from_checkpoint, "checkpoints/checkpoint.ckpt")
    config.model.metric.write_result_to_file=True
    # config.model.metric.target._target_= 'src.model.metric.AttachmentSpanMetric'
    config.root = hydra.utils.get_original_cwd()

    hydra_dir = str(os.getcwd())

    os.chdir(hydra.utils.get_original_cwd())

    datamodule = hydra.utils.instantiate(config.datamodule.target, config.datamodule, _recursive_=False)


    log.info("created datamodule")
    datamodule.setup()
    # Instantiate model, fuck hydra 1.1
    config.runner._target_ += '.load_from_checkpoint'

    model = hydra.utils.instantiate(config.runner, cfg = config, fields=datamodule.fields, datamodule=datamodule, checkpoint_path=checkpoint, _recursive_=False)
    os.chdir(hydra_dir)
    trainer = hydra.utils.instantiate(config.trainer, logger=False,replace_sampler_ddp=False, checkpoint_callback=False)
    trainer.test(model, datamodule=datamodule)


@hydra.main(config_path="configs/", config_name="config_evaluate.yaml")
def main(config):
    evaluate(config)

if __name__ == "__main__":
    main()
