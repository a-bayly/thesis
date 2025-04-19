"""
This module contains utility code for training GCDS models. It defines the model architecture and training protocol.
See subsection 4.4 in the thesis for further details. We use the chi-squared divergence for training here rather than
the KL divergence used by the authors of the method.
"""

# Generic Python libs:
import torch
from torch.nn import Module
import lightning as L
from copy import deepcopy

# Imports from NCP repo (https://github.com/CSML-IIT-UCL/NCP/tree/main/NCP):
from NCP.utils import tonp


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Generator module
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
class ForwardOperator(Module):
    def __init__(self, G_operator:Module, G_operator_kwargs:dict):

        super(ForwardOperator, self).__init__()

        self.G = G_operator(**G_operator_kwargs)

    def forward(self,z,x):
        zx = torch.cat((z,x), dim=1)
        return self.G(zx)

    def generate_sample(self,x,Z):
        """
        A function that generates samples using the trained model
        :param x: a single value for conditioning on X = x
        :param Z: a sample from the reference distribution used for generation
        :return: a sample from Y | X = x with same sample size as Z
        """

        with torch.no_grad():

            X = x.view(1, -1).repeat(Z.size(0),1)

            sample = self.forward(Z,X)

        return sample.clone().detach()
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Discrimator module
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
class Discriminator(Module):
    def __init__(self, f_operator:Module, f_operator_kwargs:dict):

        super(Discriminator, self).__init__()

        self.f = f_operator(**f_operator_kwargs)

    def forward(self,x,y):
        xy = torch.cat((x,y), dim=1)
        return self.f(xy)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
GCDS model specification:
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
class Module(L.LightningModule):
    def __init__(
            self,
            generator: ForwardOperator,
            discriminator: Discriminator,
            optimizer_fn: torch.optim.Optimizer,
            g_opt_kwargs: dict,
            d_opt_kwargs: dict,
    ):
        super(Module, self).__init__()
        self.generator = generator
        self.discriminator = discriminator

        # Important: This property activates manual optimization.
        self.automatic_optimization = False

        self._optimizer = optimizer_fn

        if "lr" not in g_opt_kwargs or "lr" not in d_opt_kwargs:
            raise ValueError("opt_kwargs must contain learning rate lr")

        self.g_opt_kwargs = deepcopy(g_opt_kwargs)
        self.d_opt_kwargs = deepcopy(d_opt_kwargs)
        self.lr = deepcopy(g_opt_kwargs).pop("lr") # For Lightning's LearningRateFinder

        def set_opt_kwargs(optimizer_kwargs):
            _tmp_opt_kwargs = deepcopy(optimizer_kwargs)
            if "lr" in _tmp_opt_kwargs:  # For Lightning's LearningRateFinder
                self.lr = _tmp_opt_kwargs.pop("lr")
                self.opt_kwargs = _tmp_opt_kwargs
            else:
                self.lr = 1e-3
                raise Warning("No learning rate specified. Using default value of 1e-3. You can specify the learning rate by passing it to the optimizer_kwargs argument.")


        self.train_loss = []
        self.val_loss = []

    def configure_optimizers(self):
        optimizer_d = self._optimizer(self.parameters(), **self.d_opt_kwargs)
        optimizer_g = self._optimizer(self.parameters(), **self.g_opt_kwargs)

        return [optimizer_d, optimizer_g], []

    def training_step(self, batch, batch_idx):

        n = 6 # discriminator
        m = 1 # generator

        if self.current_epoch == 0 and batch_idx == 0:
            self.batch = batch

        d_opt, g_opt = self.optimizers()

        X, Y, Z = batch

        if (batch_idx % (n + m)) < n:
            ##########################
            # Optimize Discriminator #
            ##########################
            d_loss = chi_score_1(X, Y, Z, self.generator, self.discriminator)

            d_opt.zero_grad()
            self.manual_backward(d_loss)
            d_opt.step()

        else:
            ######################
            # Optimize Generator #
            ######################
            g_loss = chi_score_2(X, Z, self.generator, self.discriminator)

            g_opt.zero_grad()
            self.manual_backward(g_loss)
            g_opt.step()

            ######################
            # Logging            #
            ######################

            self.log('train_loss', g_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.train_loss.append(tonp(g_loss))

    def validation_step(self, batch, batch_idx):

        X, Y, Z = batch

        v_loss = 1-chi_score_1(X,Y,Z,self.generator,self.discriminator) # lower bound for divergence score

        self.log('val_loss',v_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_loss.append(tonp(v_loss))

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Utility functions used in GCDS model training
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def random_split(*tensors,n):
    """
    Randomly splits the data into n partitions with equal size. Works with arbitrary number of input tensors.
    :param tensors: variable number of tensors
    :param n: number of partitions
    :return: List of partitions
    """
    # Ensure all tensors have the same size along the first dimension
    length = tensors[0].shape[0]
    for tensor in tensors:
        assert tensor.shape[0] == length, "All tensors must have the same size along the first dimension."

    # Adjust size to be divisible by n
    res = length % n
    if res != 0:
        tensors = tuple(tensor[:-res] for tensor in tensors)

    # Get the new length after adjustment
    length = tensors[0].shape[0]

    # Randomly shuffle the indices
    idxs = torch.randperm(length)

    # Shuffle the data
    shuffled_tensors = tuple(tensor[idxs] for tensor in tensors)

    # Determine the size of each split
    split_size = length // n

    # Create the splits
    splits = []
    for i in range(n):
        start = i * split_size
        end = (i + 1) * split_size if i != n - 1 else length
        split = tuple(tensor[start:end] for tensor in shuffled_tensors)
        splits.append(split)

    return splits
def chi_score_1(X: torch.Tensor, Y:torch.Tensor, Z:torch.Tensor, generator: ForwardOperator, discriminator: Discriminator):
    """
    This function calculates the chi-squared divergence used for backpropagation involving discriminator parameters
    """

    (X1, Y1, Z1), (X2, Y2, Z2) = random_split(X, Y, Z, n=2)

    G1 = generator.forward(Z1,X1)
    G2 = generator.forward(Z2,X2)

    dXY_1 = discriminator.forward(X1,Y1)
    dXY_2 = discriminator.forward(X2,Y2)
    dXY = torch.cat((dXY_1,dXY_2),dim=0)

    dXG_1 = discriminator.forward(X1,G1)
    dXG_2 = discriminator.forward(X2,G2)
    dXG = torch.cat((dXG_1,dXG_2),dim=0)

    loss = torch.mean(dXY.pow(2) - 2*dXG,dim=0,keepdim=True) # Note use of negative objective for maximisation

    return loss
def chi_score_2(X: torch.Tensor, Z: torch.Tensor, generator: ForwardOperator,discriminator: Discriminator):
    """
    This function calculates the component of the chi-squared divergence that depends on the generator only. Used for
    backpropagation involving generator parameters.
    """

    G = generator.forward(Z,X)
    D = discriminator.forward(X,G)

    loss = torch.mean(D, dim=0, keepdim=True)

    return loss

