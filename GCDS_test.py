"""
This module tests GCDS modelling by training models on a Linear Gaussian dataset and plotting diagnostics for samples
generated by the trained model. See chapter 4 in the thesis for discussion.
"""

# Generic Python libs:
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import statsmodels.api as sm

# Utility code written for this thesis:
from NCP_data import datasetXYZ
import GCDS_train
from LightningUtils import *

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Data generation class
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

class LinearGaussian(datasetXYZ):
    """
    This class generates data under the LinearGaussian model using the datasetXYZ utility class. The third variable Z is
    the reference variable that will be used for generative sampling on test set. We use a 5 dimensional isotropic
    multi-variate normal distribution as the reference here - see chapter 4 in the thesis for discussion on significance
    of this choice.
    """
    def __init__(self, sample_size):
        super(LinearGaussian, self).__init__(sample_size)
        self.name = "LinearGaussian"
        self.X = np.random.uniform(-1.0, 1.0, size=(sample_size, 1))
        self.Y = self.X + 0.5*np.random.normal(0.0, 1.0, (sample_size, 1))
        self.Z = np.random.normal(0.0, 1.0, (sample_size, 5))  # Reference distribution
        self.validate()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Utility function for creating diagnostic plots
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def plot_GCDS_diagnostics(model, X, Y, Z):
        """
        This function plots diagnostics for trained GCDS model
        :param model: instance of trained GCDS model
        :param X: data for X variable
        :param Y: data for Y variable
        :param Z: data for Z variable
        """

        matplotlib.use('Qt5Agg')

        quantiles = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        x_grid = torch.quantile(X, quantiles)

        for i, x in enumerate(x_grid):
            # Generate a sample from trained GCDS model:
            with torch.no_grad():
                sample = model.generate_sample(x, Z).numpy().squeeze()

            print("Sample[", i, "]: X conditioning value = ", x, "  Mean of generated sample = ", np.mean(sample),
                  " Standard deviation of generated sample=", np.std(sample))

            # Q-Q plot
            sm.qqplot((sample - np.mean(sample)) / np.std(sample), line='45')
            plt.grid(True)
            plt.title('QQ-plot of sample against quantiles of Gaussian distribution')
            plt.show()

            # Calculate ECDF for sample:
            sample_cdf_grid = np.sort(sample)
            sample_cdf_grid = (sample_cdf_grid.reshape(1, -1)).squeeze()
            sample_ecdf = np.arange(1, len(sample_cdf_grid) + 1) / len(sample_cdf_grid)

            # Plot NCP_CDF/ECDF
            plt.step(sample_cdf_grid, sample_ecdf, where="post")
            plt.xlabel('Data Points')
            plt.ylabel('ECDF')
            plt.title('Empirical Cumulative Distribution Function (ECDF) for sample')
            plt.grid(True)
            plt.show()


if __name__ == "__main__":

    global_seed = 44  # Set global seed
    np.random.seed(global_seed)

    # Dataset sizes:
    Ntrain = int(5e4)  # training set
    Nvalid = int(5e3)  # validation set
    Ntest = int(3e4)  # test set

    # Construct datasets:
    train = LinearGaussian(sample_size=Ntrain)
    valid = LinearGaussian(sample_size=Nvalid)
    test = LinearGaussian(sample_size=Ntest)

    # Apply scaler from training dataset for validation and test sets (i.e. these are OOS)
    valid.set_scaler(x_scaler=train.x_scaler, y_scaler=train.y_scaler, z_scaler=train.z_scaler)
    test.set_scaler(x_scaler=train.x_scaler, y_scaler=train.y_scaler, z_scaler=train.z_scaler)

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Training code for GENERATIVE model [Uses pytorch lightning]
    See section 4.4 in the thesis for discussion.
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    Gen_trainer = GCDS_train.LightningTrainer(_seed_=global_seed, training_data=train, validation_data=valid)
    Gen_trainer.training_kwargs['max_epochs'] = int(20)
    Gen_trainer.learning_kwargs['batch_size_train'] = 5000
    Gen_trainer.learning_kwargs['shuffle_train'] = True
    Gen_trainer.learning_kwargs['data_format'] = "X,Y,Z"
    Gen_trainer.generator_kwargs['n_hidden'] = 1  # Number of hidden layers for generator model
    Gen_trainer.generator_kwargs['layer_size'] = [30]  # Size of hidden layers for generator model
    Gen_trainer.discriminator_kwargs['n_hidden'] = 2  # Number of hidden layers for discriminator model
    Gen_trainer.discriminator_kwargs['layer_size'] = [40, 20]  # Sizes of hidden layers for discriminator model
    Gen_trainer.g_opt_kwargs['lr'] = 1e-3  # Learning Rate for generator
    Gen_trainer.d_opt_kwargs['lr'] = 1e-4  # Learning Rate for discriminator
    Gen_trainer.g_opt_kwargs['weight_decay'] = 5e-5  # Weight decay for generator
    Gen_trainer.d_opt_kwargs['weight_decay'] = 0.00  # Weight decay for discriminator (none)

    Gen_trainer.train()

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Plot diagnostics for GCDS model
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    # load model
    print("Loading model : ", Gen_trainer.paths["model_name"])
    model = torch.load(Gen_trainer.paths["model_name"]).to('cpu')
    model.eval()

    # Load test dataset
    X, Y, Z = test.fetch(scaled=False, tensor=True)

    # Plot diagnostics
    plot_GCDS_diagnostics(model, X, Y, Z)