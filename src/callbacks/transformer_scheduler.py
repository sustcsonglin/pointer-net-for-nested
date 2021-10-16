import pytorch_lightning as pl
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
import logging
log = logging.getLogger(__name__)

class TransformerLrScheduler(pl.Callback):
    def __init__(self, warmup):
        self.warmup = warmup

    def on_train_start(self, trainer, pl_module):
        for lr_scheduler in trainer.lr_schedulers:
            scheduler = lr_scheduler['scheduler']
            n_train = len(pl_module.train_dataloader())
            n_accumulate_grad = trainer.accumulate_grad_batches
            n_max_epochs = trainer.max_epochs
            num_training_steps = n_train // n_accumulate_grad * n_max_epochs
            num_warmup_steps = int(self.warmup * num_training_steps)

            if  pl_module.optimizer_cfg.scheduler_type == 'linear_warmup':
                lr_scheduler['scheduler'] = get_linear_schedule_with_warmup(scheduler.optimizer, num_warmup_steps, num_training_steps)

            elif pl_module.optimizer_cfg.scheduler_type == 'constant_warmup':
                lr_scheduler['scheduler'] = get_constant_schedule_with_warmup(scheduler.optimizer, num_warmup_steps,
                                                                            )

            log.info(f"Warm up rate:{self.warmup}")
            log.info(f"total number of training step:{num_training_steps}")
            log.info(f"number of training batches per epochs in the dataloader:{n_train}")