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
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import sys
# sys.setdefaultencoding() does not exist, here!
from omegaconf import OmegaConf, open_dict


def train(config):
    # if contains this, means we are multi-run and optuna-ing
    log.info(OmegaConf.to_container(config,resolve=True))
    config.root = hydra.utils.get_original_cwd()
    limit_train_batches = 1.0


    hydra_dir = str(os.getcwd())
    seed_everything(config.seed)
    os.chdir(hydra.utils.get_original_cwd())

    # Instantiate datamodule
    hydra.utils.log.info(os.getcwd())
    hydra.utils.log.info(f"Instantiating <{config.datamodule.target}>")
    # Instantiate callbacks and logger.
    callbacks: List[Callback] = []
    logger: List[LightningLoggerBase] = []

    # ------------------------------- debug -------------------------- #
    if config.debug is True:
        config.trainer.max_epochs = 10
        config.trainer.fast_dev_run = False
        config.trainer.gpus = 0
        # config.datamodule.debug = True
        config.datamodule.use_bert = False
        config.datamodule.use_char = True
        config.wandb = False
        # distributed = False
        # config.datamodule.suffix = '.debug'
        config.datamodule.use_cache=False
        config.datamodule.train_const += ".debug"
        config.datamodule.dev_const += ".debug"
        config.datamodule.test_const += ".debug"
        config.datamodule.use_emb = False
    else:
        pass
        # distributed = config.distributed

    # -------------------------------end debug -------------------------- #
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        config.datamodule.target, config.datamodule, _recursive_=False
    )

    log.info("created datamodule")
    datamodule.setup()
    model = hydra.utils.instantiate(config.runner, cfg = config, fields=datamodule.fields, datamodule=datamodule, _recursive_=False)

    os.chdir(hydra_dir)



# Train the model ⚡
    if "callbacks" in config:
        for _, cb_conf in config["callbacks"].items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    if config.checkpoint:
        callbacks.append(
            ModelCheckpoint(
                monitor='valid/score',
                mode='max',
                save_last=False,
                filename='checkpoint'
            )
        )
        log.info("Instantiating callback, ModelCheckpoint")


    if config.wandb:
        logger.append(hydra.utils.instantiate(config.logger))

    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger,
        replace_sampler_ddp=False,
        # accelerator='ddp' if distributed else None,
        accumulate_grad_batches=config.accumulation,
        # limit_train_batches=0.1,
        checkpoint_callback=config.checkpoint,
        # turnoff sanity check. 0 turn off, -1 all, positive number is n samples. depends on yours.
        # TODO: there is a bug which is conflits to 'write_result_to_file' TOFIX.....
        num_sanity_val_steps=0,
        # limit_train_batches=.1,
    )

    log.info(f"Starting training!")
    if config.wandb:
        logger[-1].experiment.save(str(hydra_dir) + "/.hydra/*", base_path=str(hydra_dir))

    trainer.fit(model, datamodule)
    log.info(f"Finalizing!")

    if config.wandb:
        logger[-1].experiment.save(str(hydra_dir) + "/*.log", base_path=str(hydra_dir))
        wandb.finish()

    log.info(f'hydra_path:{os.getcwd()}')


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config):
    train(config)

if __name__ == "__main__":
    main()


