"""
Code for Experiment 2a [Recovering conditional covariance]
See section 3.8.2 for details. This code generates figures 3.13-3.15.
"""

# Generic Python libs:
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FormatStrFormatter
from pathlib import Path
from torch.nn import GELU
import warnings

# Utility code written for this thesis:
from LightningUtils import *
from NCP_data import datasetXY
import NCP_train

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Preparations and settings
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
matplotlib.use('Qt5Agg')
warnings.filterwarnings("ignore", ".*set_ticklabels().*")
warnings.filterwarnings("ignore", ".*invalid value.*")


save_dir = "figures/Experiment02/"
dir_path = Path(save_dir)
dir_path.mkdir(parents=True, exist_ok=True)
_saveplots_ = False

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Generate data for experiment
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
np.random.seed(42)
class GaussianConditionalCovariance(datasetXY):
    def __init__(self, sample_size):
        """
        Generates Gaussian Conditional Covariance data - see section 3.8.2 in thesis
        :param sample_size: number data points (integer)
        """
        super(GaussianConditionalCovariance, self).__init__(sample_size)
        self.name = "GaussianConditionalCovariance"
        self.X = np.random.normal(0.0,1.0,(sample_size,2))
        self.Y = np.empty(self.X.shape)
        self.stats = np.empty((sample_size,3))

        for i in range(self.Y.shape[0]):
            s = np.abs(self.X[i,:])
            rho = 0.7*np.tanh(np.diff(self.X[i,:])).squeeze()
            self.stats[i,:2] = s
            self.stats[i,2] = rho
            C = np.array([[s[0]*s[0],s[0]*s[1]*rho],[s[0]*s[1]*rho,s[1]*s[1]]])
            self.Y[i,:] = np.random.multivariate_normal([0,0],C,size=1)

        self.validate()


# Set size of datasets:
Ntrain = int(1e6)  # training set of 100k examples
Nvalid = int(5e3)  # validation set
Ntest  = int(1e4)  # test set with 10k examples

# Create training/validation/testing datasets
train = GaussianConditionalCovariance(sample_size=Ntrain)
valid = GaussianConditionalCovariance(sample_size=Nvalid)
test = GaussianConditionalCovariance(sample_size=Ntest)

# Apply scaler from training dataset to validation/test datasets (i.e. these are OOS)
valid.set_scaler(x_scaler=train.x_scaler, y_scaler=train.y_scaler)
test.set_scaler(x_scaler=train.x_scaler, y_scaler=train.y_scaler)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Train NCP model on the data
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

NCP_trainer = NCP_train.LightningTrainer(_seed_= 51,training_data = train,validation_data = valid,d_value =200)
NCP_trainer.learning_kwargs['data_format'] = "X,Y"
NCP_trainer.learning_kwargs['scaling'] = True
NCP_trainer.learning_kwargs['hidden_layers'] = 2*[64]
NCP_trainer.training_kwargs['max_epochs'] = int(5000) # Number of epochs
NCP_trainer.learning_kwargs['batch_size_train'] = Ntrain
NCP_trainer.learning_kwargs['shuffle_train'] = False
NCP_trainer.loss_kwargs['gamma'] = 0.001
NCP_trainer.learning_kwargs['activation'] = GELU

NCP_trainer.train()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Plot evolution for loss metrics during training [generates Figure 3.13]
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def plot_loss_metrics(log_path):

    metrics_df = pd.read_csv(log_path)

    # Plot the total loss metric versus epochs
    plt.figure(figsize=(6,4))
    plt.rc('font', size=10)  # Default text size
    plt.rc('axes', titlesize=12)  # Title size
    plt.rc('lines', linewidth=1.5)  # Line thickness

    idx1 = 10
    idx2 = 4999
    plt.plot(metrics_df['epoch'][idx1:idx2], metrics_df['train_loss'][idx1:idx2], label='Training Loss', marker='o',markersize=3)
    plt.plot(metrics_df['epoch'][idx1:idx2], metrics_df['val_loss'][idx1:idx2], label='Validation Loss', marker='o',markersize=3)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss metrics vs Epoch')
    plt.legend()
    plt.grid(True)
    plt.ylim(-10,0)

    _saveplots_ = False
    if _saveplots_:
        plt.savefig(save_dir + 'CC_loss_metrics.pdf', dpi=300, bbox_inches='tight')
    else:
        plt.show()

plot_loss_metrics('lightning_logs/version_294/metrics.csv')

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Utilities
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def calculate_predicted_statistics(model,X_values,Y_sample,postprocess = 'whitening'):

    # Calculate conditional expectation of moments:
    y1 = model.conditional_expectation(X1, Y, observable=lambda y: y[:, 0].reshape(-1, 1), postprocess=postprocess)
    y2 = model.conditional_expectation(X1, Y, observable=lambda y: y[:, 1].reshape(-1, 1), postprocess=postprocess)
    y1y2 = model.conditional_expectation(X1, Y, observable=lambda y: (y[:, 0] * y[:, 1]).reshape(-1, 1),postprocess=postprocess)
    y1y1 = model.conditional_expectation(X1, Y, observable=lambda y: (y[:, 0] * y[:, 0]).reshape(-1, 1),postprocess=postprocess)
    y2y2 = model.conditional_expectation(X1, Y, observable=lambda y: (y[:, 1] * y[:, 1]).reshape(-1, 1),postprocess=postprocess)

    # Calculate full covariances:
    c1 = y1y1 - y1 * y1
    c2 = y2y2 - y2 * y2
    c3 = y1y2 - y1 * y2
    predicted_statistics = np.concatenate((c1, c2, c3), axis=1)

    # Calculate correlation:
    rho = c3 / (np.sqrt(c1) * np.sqrt(c2))

    predicted_statistics = np.concatenate((c1, c2, rho), axis=1)
    return predicted_statistics
def diagnostic_plot(predicted,ground_truth,xy_lim,name,savename=None):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6, 3.5))
    axes = ax.flatten()
    plt.rc('font', size=6)  # Default text size
    plt.rc('axes', titlesize=5)  # Title size
    plt.rc('lines', linewidth=2)  # Line thickness
    plt.rcParams['xtick.labelsize'] = 3  # Font size for x-axis tick labels
    plt.rcParams['ytick.labelsize'] = 3  # Font size for y-axis tick labels

    axes[0].scatter(ground_truth,predicted,marker='o', s=0.5)
    axes[0].plot(ground_truth,ground_truth, color='red', linewidth=2, label='Correct prediction')
    axes[0].set_xlim(xy_lim['x'][0],xy_lim['x'][1])
    axes[0].set_ylim(xy_lim['y'][0], xy_lim['y'][1])
    axes[0].grid(True)
    axes[0].legend()
    axes[0].xaxis.set_major_locator(ticker.AutoLocator())
    axes[0].set_xticklabels(axes[0].get_xticks(), fontsize=5)
    axes[0].yaxis.set_major_locator(ticker.AutoLocator())
    axes[0].set_yticklabels(axes[0].get_yticks(), fontsize=5)
    axes[0].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    axes[0].set_xlabel(f"True {name} given x-value",fontsize=6)
    axes[0].set_ylabel(f"Conditional {name} in model",fontsize=6)
    axes[0].set_title(f"Conditional {name}: estimates versus ground truth")

    err = predicted - ground_truth
    err = np.clip(err,-0.5,0.5)
    axes[1].hist(err, bins=100, color='blue', edgecolor='black', alpha=0.7)
    axes[1].set_title('Histogram of prediction errors')
    axes[1].set_xlabel('Error value',fontsize=6)
    axes[1].set_yticks([])
    axes[1].set_ylabel("")
    axes[1].set_xlim(-0.55, 0.55)
    axes[1].grid(True)

    if savename is not None:
        plt.savefig(save_dir + f"Experiment02_{savename}.pdf", dpi=300, bbox_inches='tight')
    else:
        plt.show()


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Plot prediction benchmark diagnostics [generates Figures 3.14/3.15]
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# load model:
model = torch.load(NCP_trainer.paths["model_name"])

# Fetch data:
X,Y = train.fetch(scaled = True,tensor = True) # We use this as a sample from Y for conditioning
X1,Y1 = test.fetch(scaled = True,tensor = True) # OOS data

predicted_statistics = calculate_predicted_statistics(model,X_values=X1,Y_sample=Y,postprocess='whitening')

predicted = predicted_statistics[:,0]
ground_truth = (test.stats[:,0])**2
xy_lim={'x': [0,6],'y': [0,6]}
diagnostic_plot(predicted,ground_truth,xy_lim,'variance of $Y_1$',savename=None)

predicted = predicted_statistics[:,2]
ground_truth = (test.stats[:,2])
xy_lim={'x': [-0.7,0.7],'y': [-1.3,1.3]}
diagnostic_plot(predicted,ground_truth,xy_lim,'correlation($Y_1$,$Y_2$)',savename=None)
