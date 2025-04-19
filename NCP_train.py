"""
This module contains utility code for training NCP models using PyTorch Lightning utilities. It is based on training
code from the notebook https://github.com/CSML-IIT-UCL/NCP/blob/main/NCP/examples/NCP_benchmarks.ipynb which is wrapped
as a single Python class "LightningTrainer" with added utilities.
It uses some classes and functions from the NCP repo: https://github.com/CSML-IIT-UCL/NCP/tree/main/NCP - see imports..
"""

# Generic Python libraries:
import os
import inspect
from time import perf_counter
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import GELU

# Imports from NCP repo (https://github.com/CSML-IIT-UCL/NCP/tree/main/NCP):
from NCP.nn.layers import MLP
from NCP.utils import FastTensorDataLoader
from NCP.nn.losses import CMELoss
from NCP.model import NCPOperator, NCPModule

# Utilities written for this thesis:
from LightningUtils import *
import NCP_data

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Training class using Lightning
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

class LightningTrainer():

    def __init__(self,_seed_,training_data: NCP_data.datasetXY,validation_data: NCP_data.datasetXY,d_value = 100):
        """
        Setup for training class.
        :param _seed_: Seed integer to use
        :param training_data: A dataset object from NCP_data utilities module
        :param validation_data: A dataset object from NCP_data utilities module
        :param d_value: Dimension of approximated SVE for deflated operator
        """

        self.seed = _seed_
        self.train_data = training_data
        self.valid_data = validation_data
        self.d = d_value

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
            'scaling': True,
            'loss_function': CMELoss,
            'device': 'cpu',
            'dropout': 0,
            'iterative_whitening': False,
            'hidden_layers': 2*[64],
            'activation': GELU,
            'EarlyStoppingPatience': 200,
            'data_format': "X,Y",
            'batch_size_train': len(self.train_data),
            'batch_size_valid': len(self.valid_data),
            'shuffle_train': False,
            'shuffle_valid': False
        }

        self.U_kwargs = {}
        self.V_kwargs = {}
        self.set_MLP_kwargs()

        self.optimizer = optim.Adam
        self.optimizer_kwargs = {
            'lr': 1e-3
        }

        self.loss_function = self.learning_kwargs['loss_function']
        self.loss_kwargs = {
            'mode': 'cov',
            'gamma': 1e-3
        }

        if not os.path.exists(self.paths["checkpoint"]):
            os.makedirs(self.paths["checkpoint"])

        if not os.path.exists(self.paths["logger"]):
            os.makedirs(self.paths["logger"])

    def set_MLP_kwargs(self):
        """
        Sets parameters for the feature networks
        """

        self.U_kwargs = {
            'input_shape': self.train_data.x_dim,
            'output_shape': self.d,
            'n_hidden': len(self.learning_kwargs['hidden_layers']),
            'layer_size': self.learning_kwargs['hidden_layers'],
            'dropout': self.learning_kwargs['dropout'],
            'iterative_whitening': self.learning_kwargs['iterative_whitening'],
            'activation': self.learning_kwargs['activation']
        }

        self.V_kwargs = {
            'input_shape': self.train_data.y_dim,
            'output_shape': self.d,
            'n_hidden': len(self.learning_kwargs['hidden_layers']),
            'layer_size': self.learning_kwargs['hidden_layers'],
            'dropout': self.learning_kwargs['dropout'],
            'iterative_whitening': self.learning_kwargs['iterative_whitening'],
            'activation': self.learning_kwargs['activation']
        }

    def set_paths(self):
        """
        Sets paths for saving log files, models and checkpoints
        """

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
            "model_name": "models/" + file_name + "/NCP_" + self.train_data.name + "_seed_" + str(self.seed) +".pt",
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
        """
        Trains model - records loss values - saves model checkpoints
        Implements early stopping procedure
        """

        if os.path.isfile(self.paths["model_name"]):
            print("NCP model: '" + self.paths["model_name"] + "' has already been trained")
            return

        L.seed_everything(self.seed)

        NCP_model = NCPOperator(U_operator=MLP, V_operator=MLP, U_operator_kwargs=self.U_kwargs, V_operator_kwargs=self.V_kwargs)

        NCP_module = NCPModule(
            NCP_model,
            self.optimizer,
            self.optimizer_kwargs,
            self.learning_kwargs['loss_function'],
            self.loss_kwargs
        )

        early_stop = EarlyStopping(monitor="val_loss", patience=self.learning_kwargs["EarlyStoppingPatience"], mode="min")

        checkpoint_callback = CustomModelCheckpoint(save_top_k=1, monitor="val_loss", mode="min", dirpath=self.paths["checkpoint"])

        trainer = L.Trainer(**self.training_kwargs,callbacks=[LitProgressBar(), early_stop, checkpoint_callback])

        data_format = self.learning_kwargs["data_format"]
        scaling = self.learning_kwargs["scaling"]
        X_train, Y_train = self.train_data.fetch(format = data_format,scaled=scaling, tensor=True)
        X_valid, Y_valid = self.valid_data.fetch(format = data_format,scaled=scaling, tensor=True)

        device = self.learning_kwargs["device"]
        train_dataloader = FastTensorDataLoader(X_train.to(device), Y_train.to(device), batch_size = self.learning_kwargs["batch_size_train"],shuffle = self.learning_kwargs["shuffle_train"])
        valid_dataloader = FastTensorDataLoader(X_valid.to(device), Y_valid.to(device), batch_size = self.learning_kwargs["batch_size_valid"],shuffle = self.learning_kwargs["shuffle_valid"])


        start = perf_counter()
        trainer.fit(NCP_module, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
        end = perf_counter()
        print(f'Training time: {end - start}')

        best_model = torch.load(self.paths["checkpoint"] + '/best_model.pt').to('cpu')
        print(checkpoint_callback.best_model_path)
        torch.save(best_model,self.paths["model_name"])


