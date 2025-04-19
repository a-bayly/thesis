"""
Utility module for implementing Lightning module functionality
"""

import torch
from tqdm import tqdm
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Lightning module functionality
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Class for preventing the wall of text
class LitProgressBar(L.pytorch.callbacks.TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = tqdm(disable=True,)
        return bar

    def init_train_tqdm(self):
        bar = tqdm()
        bar.dynamic_ncols = False
        bar.ncols = 0
        bar.bar_format ='{desc} [{rate_fmt}{postfix}]'
        return bar


class MyEarlyStopping(EarlyStopping):
    def on_train_end(self, trainer, pl_module):
        # instead, do it at the end of training loop
        self._run_early_stopping_check(trainer)


class CustomModelCheckpoint(ModelCheckpoint):
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        X, Y = trainer.model.batch
        trainer.model.model._compute_data_statistics(X, Y)
        torch.save(trainer.model.model, trainer.checkpoint_callback.dirpath + '/best_model.pt')


class CustomModelCheckpoint1(ModelCheckpoint):
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        torch.save(trainer.model.model, trainer.checkpoint_callback.dirpath + '/best_model.pt')