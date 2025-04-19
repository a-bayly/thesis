"""
Data utilities for generating training/validation/test datasets for NCP modelling. This code was written specifically
for the purposes of the thesis.
"""

import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from abc import ABC, abstractmethod

def convert_to_tensor(input_array):
    """
    Converts a NumPy array to a torch.tensor if the input is NumPy array
    ELSE if the input is a torch.tensor then returns the input unmodified
    ELSE throw exception

    Parameters:
    - input_array: A NumPy array or a PyTorch tensor.

    Returns:
    - A PyTorch tensor.
    """
    if isinstance(input_array, np.ndarray):
        # Convert NumPy array to PyTorch tensor
        return torch.from_numpy(input_array)
    elif isinstance(input_array, torch.Tensor):
        # Already a PyTorch tensor
        return input_array
    else:
        raise TypeError("Input should be a NumPy array or a PyTorch tensor.")
class datasetXY(Dataset, ABC):
    """
    This is an abstract base class for generating datasets with standard utilities. This class caters the situation where
    there are samples for two variables X and Y of equal size.
    """

    def __init__(self,sample_size):
        super(datasetXY, self).__init__()
        self.sample_size = sample_size
        self.name = "NONE"
        self.x_scaler,self.x_dim,self.X = None,None,None
        self.y_scaler,self.y_dim,self.Y = None,None,None

    def __len__(self):
        return self.sample_size

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]

        return x,y

    def dim_X(self):
        return self.x_dim

    def dim_Y(self):
        return self.y_dim

    def fetch(self,format=None, scaled = False,tensor = False):

        x = self.X
        y = self.Y

        if scaled:
            x = self.x_scaler.transform(x)
            y = self.y_scaler.transform(y)

        if tensor:
            x = convert_to_tensor(x).float()
            y = convert_to_tensor(y).float()

        return x,y

    def set_scaler(self,x_scaler: StandardScaler, y_scaler: StandardScaler):
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler

    def validate(self):
        # Perform validation checks on data
        if self.X is None or self.Y is None: raise ValueError("Data not initialized")
        if len(self.X.shape) != 2 or len(self.Y.shape) != 2: raise ValueError("Training data for X and Y needs to be a 2D array")
        if self.X.shape[0] != self.sample_size or self.Y.shape[0] != self.sample_size: raise ValueError("Data for X and Y not conformal with sample size")
        self.x_dim = self.X.shape[1]
        self.y_dim = self.Y.shape[1]
        self.x_scaler = StandardScaler()
        self.x_scaler.fit_transform(self.X)
        self.y_scaler = StandardScaler()
        self.y_scaler.fit_transform(self.Y)

class datasetXYZ(Dataset, ABC):
    """
    This is an abstract base class for generating datasets with standard utilities. This class caters the situation where
    there are samples for three variables X,Y, and Z of equal size.
    """

    def __init__(self,sample_size):
        super(datasetXYZ, self).__init__()
        self.sample_size = sample_size
        self.name = "NONE"
        self.x_scaler,self.x_dim,self.X = None,None,None
        self.y_scaler,self.y_dim,self.Y = None,None,None
        self.z_scaler,self.z_dim,self.Z = None,None,None

    def __len__(self):
        return self.sample_size

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]
        z = self.Z[idx]

        return x,y,z

    def dim_X(self):
        return self.x_dim

    def dim_Y(self):
        return self.y_dim

    def dim_Z(self):
        return self.z_dim

    def fetch(self,format: str = None, scaled = False,tensor = False):

        x = self.X
        y = self.Y
        z = self.Z

        if scaled:
            x = self.x_scaler.transform(x)
            y = self.y_scaler.transform(y)
            z = self.z_scaler.transform(z)

        if tensor:
            x = convert_to_tensor(x).float()
            y = convert_to_tensor(y).float()
            z = convert_to_tensor(z).float()

        if format == "(Z,X),Y":
            zx = torch.cat((z,x),dim=1)
            return zx,y
        elif format == "X,Y":
            return x,y
        elif format == "Z,X":
            return z,x
        elif format == "X":
            return x
        elif format == "Y":
            return y
        elif format == "Z":
            return z
        else: # DEFAULT
            return x,y,z

    def set_scaler(self,x_scaler: StandardScaler, y_scaler: StandardScaler,z_scaler: StandardScaler):
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.z_scaler = z_scaler

    def validate(self):
        # Perform validation checks
        if self.X is None or self.Y is None or self.Z is None: raise ValueError("Data not initialized")
        if len(self.X.shape) != 2 or len(self.Y.shape) != 2 or len(self.Z.shape) != 2: raise ValueError("Training data needs to be a 2D array for each variable")
        if self.X.shape[0] != self.sample_size or self.Y.shape[0] != self.sample_size or self.Z.shape[0] != self.sample_size: raise ValueError("Data not conformal with sample size")

        self.x_dim = self.X.shape[1]
        self.y_dim = self.Y.shape[1]
        self.z_dim = self.Z.shape[1]

        self.x_scaler = StandardScaler()
        self.x_scaler.fit_transform(self.X)
        self.y_scaler = StandardScaler()
        self.y_scaler.fit_transform(self.Y)
        self.z_scaler = StandardScaler()
        self.z_scaler.fit_transform(self.Z)

