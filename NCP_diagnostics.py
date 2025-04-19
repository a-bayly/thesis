"""
The code in this file generates plots 3.2-3.8 in chapter 3 of the thesis.
These are diagnostic plots that are useful for understanding how the NCP method operates.
The data generating model used here is a basic Linear-Gaussian model (see Figure 3.2 in thesis for definition) and
the class definition for LinearGaussian for its implementation.

Some of this code is adapted from the codebase for the NCP paper which can be found at https://github.com/CSML-IIT-UCL/NCP
The authors of that codebase are Gregoire Pacreau, Giacomo Turri, Pietro Novelli.
"""

# Standard libraries:
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Modules from repository created by authors of NCP paper:
from NCP.utils import frnp
import NCP.cde_fork.density_simulation
from NCP.cdf import compute_marginal, integrate_pdf, smooth_cdf
from NCP.metrics import kolmogorov_smirnov

# Utilities written for the thesis (uses modifications of code from NCP repository in places)
from NCP_data import datasetXY
import NCP_train
from LightningUtils import *

# Other standard imports:
from matplotlib.ticker import FormatStrFormatter
from pathlib import Path
import matplotlib
import warnings

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Preparations and settings
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

matplotlib.use('Qt5Agg')
warnings.filterwarnings("ignore", ".*does not have many workers.*")

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Data generation class
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
np.random.seed(42)
class LinearGaussian(datasetXY):
    """
    This class generates data under the LinearGaussian model using the datasetXY utility class
    """
    def __init__(self, sample_size):
        """
        Data X,Y where X ~ U(-1,1) and Y = X + 0.5*Z where Z ~ N(0,1)
        :param sample_size: integer number of data points to generate
        """
        super(LinearGaussian, self).__init__(sample_size)
        self.name = "LinearGaussian"
        self.X = np.random.uniform(-1.0,1.0,size = (sample_size,1))
        self.Y = self.X + 0.5 * np.random.normal(0.0,1.0,(sample_size,1))
        self.validate()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Plot evolution for loss metrics during training [generates Figure 3.2]
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def plot_loss_metrics(log_path):

    metrics_df = pd.read_csv(log_path)

    # Plot the total loss metric versus epochs
    plt.figure(figsize=(6,4))
    plt.rc('font', size=10)  # Default text size
    plt.rc('axes', titlesize=12)  # Title size
    plt.rc('lines', linewidth=1.5)  # Line thickness

    plt.plot(metrics_df['epoch'][:300], metrics_df['train_loss'][:300], label='Training Loss', marker='o',markersize=3)
    plt.plot(metrics_df['epoch'][:300], metrics_df['val_loss'][:300], label='Validation Loss', marker='o',markersize=3)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss metrics vs Epoch')
    plt.legend()
    plt.grid(True)

    if _saveplots_:
        plt.savefig(save_dir + 'loss_metrics_d50.pdf', dpi=300, bbox_inches='tight')
    else:
        plt.show()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Plot diagnostics for NCP model [generates Figures 3.3/3.4]
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# This function was adapted from code in the notebook:
# https://github.com/CSML-IIT-UCL/NCP/blob/main/NCP/examples/NCP_benchmarks.ipynb
def plot_NCP_diagnostics(model,X,Y,density,inverse_scale=True,postprocess = None):

    # Three X values for conditioning:
    x_grid = np.percentile(X, [10, 50, 90])

    # Linspace of Y values
    p1, p99 = np.percentile(Y, [0.5, 99.5])
    y_grid, step = np.linspace(p1, p99, num=1000, retstep=True)
    y_grid = frnp(y_grid.reshape(-1, 1)) # as tensor

    # Marginal - from NCP.cdf module
    p_y = compute_marginal(bandwidth='scott').fit(Y)

    # Create fig with two charts  - pdf,cdf fit side by side
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6,3.5))
    axes = ax.flatten()
    plt.rc('font', size=5)  # Default text size
    plt.rc('axes', titlesize=4)  # Title size
    plt.rc('lines', linewidth=5.0)  # Line thickness

    # Fetch default color cycle for Matplotlib
    colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # Storage for KS metric by X value and model
    KS = np.zeros(len(x_grid))

    xscaler = train.x_scaler
    yscaler = train.y_scaler

    # For each X conditioning value
    for i, xi in enumerate(x_grid):

        print("i:",i,"xi:",xi)

        xi = xi.reshape(1, -1)

        # Fetch model pdf for y|X = xi (both y values and probability that Y = y | X = xi)
        fys, pred_pdf = model.pdf(frnp(xi), frnp(y_grid), postprocess=postprocess, p_y=p_y)
        fys = fys.reshape(-1, 1)

        # True pdf using code from CDE module
        x = xi
        y = y_grid
        if inverse_scale:
            x = xscaler.inverse_transform(x)
            y = yscaler.inverse_transform(y)
        x = np.tile(x, (len(y_grid), 1))

        true_pdf = density.pdf(x,y).squeeze()

        # Plot true pdf and predicted pdf on Chart 1
        pred_pdf_plot = pred_pdf.squeeze()
        if inverse_scale:
            fys = yscaler.inverse_transform(fys)
            pred_pdf_plot = pred_pdf_plot/yscaler.scale_

        axes[0].plot(fys, true_pdf, color=colours[i], linewidth=0.8, linestyle='--')
        xlabel_value = xscaler.inverse_transform(x_grid[i].reshape(-1,1)).squeeze()
        axes[0].plot(fys, pred_pdf_plot, color=colours[i],linewidth=0.5, alpha=0.5,label="Y | " +  f'x = {xlabel_value:.2f}')

        # Fetch predicted cdf using integrate_pdf from NCP.cdf module
        pred_cdf = integrate_pdf(pred_pdf, y_grid)
        smoothed_cdf = smooth_cdf(y_grid, pred_cdf)
        # Get true cdf using code from cde_fork
        true_cdf = density.cdf(x,y).squeeze()

        KS[i] = kolmogorov_smirnov(true_cdf, pred_cdf, y_grid)
        axes[1].plot(fys, true_cdf, color=colours[i], linewidth=0.8, linestyle='--')
        axes[1].plot(fys, smoothed_cdf, color=colours[i], linewidth=0.5,alpha=0.5,label="CDF for Y | " +  f'x = {xlabel_value:.2f}' + f" [KS = {KS[i]:.3f}]" )

    axes[0].set_xmargin(0)
    axes[0].set_ylabel(r"$\mathbf{\mathrm{f}}_{Y\mid X}\;\left[y \mid x\right]$")
    axes[0].set_xlabel("y value")
    axes[0].tick_params(axis="x", pad=2, labelsize=6)
    axes[0].tick_params(axis="y", pad=1, labelsize=6)
    axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    axes[0].legend(loc = "upper left",framealpha=1.0,fontsize= 4)
    axes[0].grid(True)
    axes[0].set_title("Modelled conditional densities versus ground truth",fontsize=7)

    axes[1].set_xmargin(0)
    axes[1].set_ylabel(r"$\mathbf{\mathrm{P}}\;\left[Y\leq y \mid X = x\right]$")
    axes[1].set_xlabel("y value")
    axes[1].tick_params(axis="x", pad=2, labelsize=6)
    axes[1].tick_params(axis="y", pad=1, labelsize=6)
    axes[1].legend(loc = "upper left",framealpha=1.0,fontsize=4)
    axes[1].grid(True)
    axes[1].set_title("Modelled conditional CDFs versus ground truth",fontsize=7)

    plt.tight_layout()

    if _saveplots_:
        if postprocess is None:
            postprocess_tag = "no_postprocess"
        else:
            postprocess_tag = postprocess

        plt.savefig(save_dir + "LG01_cdf_comparison_" + postprocess_tag + ".pdf", dpi=300, bbox_inches='tight')
    else:
        plt.show()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Feature correlation [generates Figure 3.5]
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def cov_plot(model,X,Y):
    plt.figure(figsize=(6, 4))
    plt.rc('font', size=6)  # Default text size
    plt.rc('axes', titlesize=6)  # Title size

    # Correlation matrix for learned features with no postprocessing:
    sigma0 = torch.sqrt(torch.exp(-model.S.weights ** 2)).detach().numpy()
    descending_indices = np.argsort(sigma0)[::-1]
    u0 = model.U(X).detach().numpy()
    cov0 = np.corrcoef(u0[:,descending_indices], rowvar=False)

    # Correlation matrix for learned features with whitening:
    u1,_,_ = model.postprocess_UV(X, Y, postprocess='whitening')
    u1 = u1.detach().numpy()
    cov1 = np.corrcoef(u1, rowvar=False)

    # Heatmap for correlation matrix for learned features with no postprocessing:
    plt.subplot(1, 2, 1)
    im = plt.imshow(cov0, cmap='coolwarm', interpolation='nearest')  # Display the matrix as a heatmap
    plt.colorbar(label='Correlation',shrink = 0.7)
    plt.title("Feature correlation (no post-processing)")
    plt.xticks([])  # Remove x-tick labels
    plt.yticks([])  # Remove y-tick labels
    plt.xlabel("d = 200")

    # Heatmap for correlation matrix for learned features with whitening:
    plt.subplot(1, 2, 2)
    plt.imshow(cov1, cmap='coolwarm', interpolation='nearest')  # Display the matrix as a heatmap
    plt.colorbar(label='Correlation', shrink=0.7)
    plt.title("Feature correlation after whitening")
    plt.xticks([])  # Remove x-tick labels
    plt.yticks([])  # Remove y-tick labels
    plt.xlabel("d (modified) = 6")

    if _saveplots_:
        plt.savefig(save_dir + 'feature_correlation.pdf', dpi=300, bbox_inches='tight')
    else:
        plt.show()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Scree plot for learned singular values [generates Figure 3.6]
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def scree_plot(model,X,Y):
    plt.figure(figsize=(6, 3))
    plt.rc('font', size=6)  # Default text size
    plt.rc('axes', titlesize=6)  # Title size
    plt.rc('lines', linewidth=1.5)  # Line thickness

    sigma0 = torch.sqrt(torch.exp(-model.S.weights ** 2)).detach().numpy()
    sigma0 = np.sort(sigma0)[::-1]
    plt.subplot(1, 2, 1)
    plt.plot(sigma0,marker='o',markersize=3)
    plt.grid(True)
    plt.title("Singular values (no post-processing)")

    _,sigma1,_ = model.postprocess_UV(X, Y, postprocess='whitening')
    sigma1 = sigma1.detach().numpy()
    sigma1 = np.sort(sigma1)[::-1]
    plt.subplot(1, 2, 2)
    plt.plot(sigma1, marker='o', markersize=5,color='red')
    plt.grid(True)
    plt.title("Singular values after whitening")

    if _saveplots_:
        plt.savefig(save_dir + 'singular_values.pdf', dpi=300, bbox_inches='tight')
    else:
        plt.show()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Plot learned features [generates Figure 3.7]
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def feature_plot(model):
    plt.figure(figsize=(6, 3))
    plt.rc('font', size=6)  # Default text size
    plt.rc('axes', titlesize=6)  # Title size
    plt.rc('lines', linewidth=0.8)  # Line thickness

    sigma0 = torch.sqrt(torch.exp(-model.S.weights ** 2)).detach().numpy()
    descending_indices = np.argsort(sigma0)[::-1]
    x_values = torch.linspace(-1.73,1.73,steps=1000).reshape(-1,1)
    utheta0 = model.U(x_values).detach().numpy()
    utheta0= utheta0[:,descending_indices]

    utheta1,_,_ = model.postprocess_UV(x_values,x_values, postprocess='whitening')
    utheta1 = utheta1.detach().numpy()

    plt.subplot(1, 2, 1)
    plt.plot(x_values,utheta0[:,:6])
    plt.title("Learned features " + r'$u_i^{\theta}$' + " (no post-processing)")
    plt.xlabel("X value")
    plt.ylabel(r'$u_i^{\theta}(x)$')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(x_values, utheta1[:, :6])
    plt.title("Learned features " + r'$u_i^{\theta}$' + " (after whitening)")
    plt.xlabel("X value")
    plt.grid(True)

    if _saveplots_:
        plt.savefig(save_dir + 'feature_examples.pdf', dpi=300, bbox_inches='tight')
    else:
        plt.show()


if __name__ == "__main__":

    # General settings:
    _saveplots_ = False
    save_dir = "figures/diagnostics01/"
    dir_path = Path(save_dir)
    dir_path.mkdir(parents=True, exist_ok=True)

    # Dataset sizes:
    Ntrain = int(5e4)  # training set
    Nvalid = int(5e3)  # validation set
    Ntest = int(5e4)  # test set

    # Construct datasets:
    train = LinearGaussian(sample_size=Ntrain)
    valid = LinearGaussian(sample_size=Nvalid)
    test = LinearGaussian(sample_size=Ntest)

    # Apply scaler from training dataset for validation and test sets (i.e. these are OOS)
    valid.set_scaler(x_scaler=train.x_scaler, y_scaler=train.y_scaler)
    test.set_scaler(x_scaler=train.x_scaler, y_scaler=train.y_scaler)

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Training settings for NCP network [Uses pytorch lightning]
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    NCP_trainer = NCP_train.LightningTrainer(_seed_=42, training_data=train, validation_data=valid, d_value=200)
    NCP_trainer.learning_kwargs['data_format'] = "X,Y"
    NCP_trainer.learning_kwargs['scaling'] = True  # This scaling is important for learning (noisier without it)
    NCP_trainer.learning_kwargs['hidden_layers'] = 2 * [64]  # 2 hidden layers with 64 nodes each
    NCP_trainer.training_kwargs['max_epochs'] = int(1000)  # number of epochs for training
    NCP_trainer.learning_kwargs['batch_size_train'] = Ntrain  # batch size used with training set
    NCP_trainer.learning_kwargs['shuffle_train'] = False  # Shuffle training batches
    NCP_trainer.loss_kwargs['gamma'] = 0.001  # mixing parameter for loss function (see NCP paper)

    NCP_trainer.train()

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Diagnostic plots
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    # [generate Figure 3.2]
    plot_loss_metrics('lightning_logs/version_276/metrics.csv')

    # [generate Figures 3.3/3.4]
    # load model
    model_0 = torch.load(NCP_trainer.paths["model_name"])

    density_params = {'ndim_x': 1, 'mu': 0, 'mu_slope': 1.0, 'std': 0.5, 'std_slope': 0.0, 'random_seed': 42}
    density = NCP.cde_fork.density_simulation.LinearGaussian(**density_params)

    X, Y = test.fetch(scaled=True, tensor=True)
    plot_NCP_diagnostics(model=model_0, X=X, Y=Y, density=density, inverse_scale=True, postprocess=None)
    plot_NCP_diagnostics(model=model_0, X=X, Y=Y, density=density, inverse_scale=True, postprocess="whitening")

    # [generate Figure 3.5]
    cov_plot(model_0, X, Y)

    # [generate Figure 3.6]
    scree_plot(model_0, X, Y)

    # [generate Figure 3.7]
    feature_plot(model_0)

    # [generate Figure 3.8]
    # Retrain with loss mixing parameter gamma set to 0.1:
    NCP_trainer = NCP_train.LightningTrainer(_seed_=43, training_data=train, validation_data=valid, d_value=200)
    NCP_trainer.learning_kwargs['data_format'] = "X,Y"
    NCP_trainer.learning_kwargs['scaling'] = True  # This scaling is important for learning (noisier without it)
    NCP_trainer.learning_kwargs['hidden_layers'] = 2 * [64]  # 2 hidden layers with 64 nodes each
    NCP_trainer.training_kwargs['max_epochs'] = int(1000)  # number of epochs for training
    NCP_trainer.learning_kwargs['batch_size_train'] = Ntrain  # batch size used with training set
    NCP_trainer.learning_kwargs['shuffle_train'] = False  # Shuffle training batches
    NCP_trainer.loss_kwargs['gamma'] = 0.1  # mixing parameter for loss function (see NCP paper)

    NCP_trainer.train()

    model_1 = torch.load(NCP_trainer.paths["model_name"])

    plot_NCP_diagnostics(model=model_1, X=X, Y=Y, density=density, inverse_scale=True, postprocess=None)