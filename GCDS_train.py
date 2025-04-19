"""
This module contains utility code for training GCDS models using pytorch Lightning library
"""

# Generic Python libs:
import os
import torch
import inspect
from time import perf_counter
import torch.optim as optim
from torch.nn import ReLU
import lightning as L
from LightningUtils import LitProgressBar
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint

# Imports from NCP repo (https://github.com/CSML-IIT-UCL/NCP/tree/main/NCP):
from NCP.nn.layers import MLP
from NCP.utils import FastTensorDataLoader

# Utility code written for this thesis:
import NCP_data
import GCDS_model


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Training class using Lightning
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

class CustomModelCheckpoint(ModelCheckpoint):
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        torch.save(trainer.model.generator, trainer.checkpoint_callback.dirpath + '/best_model.pt')
class EarlyStoppingBelowThreshold(L.Callback):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def on_validation_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get('val_loss')
        if val_loss and val_loss < self.threshold:
            print(f"Validation loss {val_loss} is below the threshold {self.threshold}. Stopping training.")
            trainer.should_stop = True

class LightningTrainer():

    def __init__(self,_seed_, training_data: NCP_data.datasetXYZ,validation_data: NCP_data.datasetXYZ):

        self.seed = _seed_
        self.train_data = training_data
        self.valid_data = validation_data

        self.paths = {}
        self.set_paths()

        self.training_kwargs = {
            'accelerator': 'auto',
            'max_epochs': int(1e3),
            'log_every_n_steps': 1,
            'enable_progress_bar': True,
            'devices': 1,
            'enable_checkpointing': True,
            'num_sanity_val_steps': 0,
            'check_val_every_n_epoch': 10,
            'enable_model_summary': True,
        }

        self.learning_kwargs = {
            'device': 'cpu',
            'dropout': 0,
            'iterative_whitening': False,
            'hidden_layers': 2*[64],
            'activation': ReLU,
            'EarlyStoppingPatience': 200,
            'batch_size_train': len(self.train_data),
            'batch_size_valid': len(self.valid_data),
            'shuffle_train': False,
            'shuffle_valid': False
        }

        self.generator_kwargs = {
            'input_shape': self.train_data.x_dim + self.train_data.z_dim,
            'output_shape': self.train_data.y_dim,
            'n_hidden': len(self.learning_kwargs['hidden_layers']),
            'layer_size': self.learning_kwargs['hidden_layers'],
            'dropout': self.learning_kwargs['dropout'],
            'activation': self.learning_kwargs['activation']
        }

        self.discriminator_kwargs = {
            'input_shape': self.train_data.x_dim + self.train_data.y_dim,
            'output_shape': 1,
            'n_hidden': len(self.learning_kwargs['hidden_layers']),
            'layer_size': self.learning_kwargs['hidden_layers'],
            'dropout': self.learning_kwargs['dropout'],
            'activation': self.learning_kwargs['activation']
        }

        self.optimizer = optim.Adam
        self.d_opt_kwargs = {
            'lr': 1e-4,
            'weight_decay': 0.00
        }
        self.g_opt_kwargs = {
            'lr': 1e-3,
            'weight_decay': 5e-6
        }

        if not os.path.exists(self.paths["checkpoint"]):
            os.makedirs(self.paths["checkpoint"])

        if not os.path.exists(self.paths["logger"]):
            os.makedirs(self.paths["logger"])

    def set_paths(self):

        frame = inspect.currentframe()

        # Move up two frames in the stack to reach the caller of the class instantiation
        # - frame.f_back is the call to __init__
        # - frame.f_back.f_back is the actual instantiation point
        instantiation_frame = frame.f_back.f_back

        # Get the file name from the frame's code object
        file_name = instantiation_frame.f_code.co_filename
        file_name = os.path.basename(file_name)
        if file_name.endswith('.py'):
            file_name = file_name[:-3]

        self.paths = {
            "model": "models/" + file_name,
            "model_name": "models/" + file_name + "/GCDS_" + self.train_data.name + "_seed_" + str(self.seed) +".pt",
            "checkpoint": "checkpoints/NCP",
            "logger": "lightning_logs/NCP"
        }

        if not os.path.exists(self.paths["model"]):
            os.makedirs(self.paths["model"])

        if not os.path.exists(self.paths["checkpoint"]):
            os.makedirs(self.paths["checkpoint"])

        if not os.path.exists(self.paths["logger"]):
            os.makedirs(self.paths["logger"])

    def train(self):

        if os.path.isfile(self.paths["model_name"]):
            print("GCDS model: '" + self.paths["model_name"] + "' has already been trained")
            return

        L.seed_everything(self.seed)

        generator = GCDS_model.ForwardOperator(G_operator=MLP, G_operator_kwargs=self.generator_kwargs)

        discriminator = GCDS_model.Discriminator(f_operator=MLP, f_operator_kwargs=self.discriminator_kwargs)

        module = GCDS_model.Module(
            generator=generator,
            discriminator=discriminator,
            optimizer_fn=self.optimizer,
            g_opt_kwargs = self.g_opt_kwargs,
            d_opt_kwargs = self.d_opt_kwargs
        )

        early_stop = EarlyStopping(monitor="val_loss", patience=self.learning_kwargs["EarlyStoppingPatience"], mode="min")

        early_stopping_callback = EarlyStoppingBelowThreshold(threshold=0.001)

        checkpoint_callback = CustomModelCheckpoint(save_top_k=1, monitor="val_loss", mode="min", dirpath=self.paths["checkpoint"])

        trainer = L.Trainer(**self.training_kwargs,callbacks=[LitProgressBar(),early_stopping_callback, early_stop, checkpoint_callback])

        X_train, Y_train, Z_train = self.train_data.fetch(format = None,scaled=False, tensor=True)
        X_valid, Y_valid, Z_valid = self.valid_data.fetch(format = None,scaled=False, tensor=True)

        device = self.learning_kwargs["device"]
        X_train.to(device)
        Y_train.to(device)
        Z_train.to(device)
        X_valid.to(device)
        Y_valid.to(device)
        Z_valid.to(device)

        train_dataloader = FastTensorDataLoader(X_train, Y_train, Z_train, batch_size = self.learning_kwargs["batch_size_train"],shuffle = self.learning_kwargs["shuffle_train"])
        valid_dataloader = FastTensorDataLoader(X_valid, Y_valid, Z_valid, batch_size = self.learning_kwargs["batch_size_valid"],shuffle = self.learning_kwargs["shuffle_valid"])

        start = perf_counter()
        trainer.fit(module, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
        end = perf_counter()
        print(f'Training time: {end - start}')

        #best_model = torch.load(self.paths["checkpoint"] + '/best_model.pt').to('cpu')
        #print(checkpoint_callback.best_model_path)
        #torch.save(best_model,self.paths["model_name"])
        torch.save(module.generator, self.paths["model_name"])

