from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint, ProgressBar
from tqdm import tqdm

class PrettyProgressBar(ProgressBar):
    """Good print wrapper."""
    def __init__(self, refresh_rate: int, process_position: int):
        super().__init__(refresh_rate=refresh_rate, process_position=process_position)

    def init_sanity_tqdm(self) -> tqdm:
        bar = tqdm(desc='Validation sanity check',
                   position=self.process_position,
                   disable=self.is_disabled,
                   leave=False,
                   ncols=120,
                   ascii=True)
        return bar

    def init_train_tqdm(self) -> tqdm:
        bar = tqdm(desc='Training',
                   initial=self.train_batch_idx,
                   position=self.process_position,
                   disable=self.is_disabled,
                   leave=True,
                   smoothing=0,
                   ncols=120,
                   ascii=True)
        return bar



    def init_validation_tqdm(self) -> tqdm:
        bar = tqdm(disable=True)
        return bar

    def on_epoch_start(self, trainer, pl_module):
        super().on_epoch_start(trainer, pl_module)
        self.main_progress_bar.set_description(f'E{trainer.current_epoch}|train')

    def on_validation_start(self, trainer, pl_module):
        super().on_validation_start(trainer, pl_module)
        self.main_progress_bar.set_description(f'E{trainer.current_epoch}|val')

    def on_test_start(self, trainer, pl_module):
        super().on_test_start(trainer, pl_module)
        self.main_progress_bar.set_description(f'E{trainer.current_epoch}|test')

