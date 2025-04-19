"""
Code for Experiment 1a [Polar transform with multiplicative noise]
See section 3.8.1 for details. This code generates figures 3.9-3.11.
"""

# Generic Python libs:
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.stats import chi2
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

# Utility code written for this thesis:
from LightningUtils import *
from NCP_data import datasetXY
import NCP_train

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Preparations and settings
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
matplotlib.use('Qt5Agg')

save_dir = "figures/Experiment01/"
dir_path = Path(save_dir)
dir_path.mkdir(parents=True, exist_ok=True)
_saveplots_ = False

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Generate data for experiment
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
np.random.seed(42)
def polar_transform(X,_mean_=False):
    if _mean_ == False:
        r = X[:,0].reshape(-1,1)
        theta = 2 * np.pi * X[:, 1].reshape(-1, 1)
        XY = r*np.concatenate((np.cos(theta),np.sin(theta)),axis=1)
    else:
        r = np.ones_like(X[:,0]).reshape(-1,1)
        theta = 2 * np.pi * X[:, 1].reshape(-1, 1)
        XY = r * np.concatenate((np.cos(theta), np.sin(theta)), axis=1)

    return XY

class CircularDistribution(datasetXY):
    def __init__(self, sample_size):
        super(CircularDistribution, self).__init__(sample_size)
        self.name = "CircularDistribution"
        self.X = np.random.uniform(0.0,1.0,(sample_size,10))
        noise = np.random.normal(0.0,1.0,(sample_size,10))
        noise = 0.1*np.sum(noise**2,axis=1,keepdims=True)
        U = np.concatenate((noise,self.X),axis=1)
        self.Y = polar_transform(U)
        self.validate()


Ntrain = int(5e4)  # training set
Nvalid = int(5e3)  # validation set
Ntest  = int(5e4)  # test set

train = CircularDistribution(sample_size=Ntrain)
valid = CircularDistribution(sample_size=Nvalid)
test = CircularDistribution(sample_size=Ntest)

# Apply scaler from training dataset to validation/test datasets (i.e. these are OOS)
valid.set_scaler(x_scaler=train.x_scaler, y_scaler=train.y_scaler)
test.set_scaler(x_scaler=train.x_scaler, y_scaler=train.y_scaler)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Train NCP model on the data
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

NCP_trainer = NCP_train.LightningTrainer(_seed_= 47,training_data = train,validation_data = valid,d_value = 200)
NCP_trainer.learning_kwargs['data_format'] = "X,Y"
NCP_trainer.learning_kwargs['scaling'] = True
NCP_trainer.learning_kwargs['hidden_layers'] = 2*[64]
NCP_trainer.training_kwargs['max_epochs'] = int(3000)
NCP_trainer.learning_kwargs['batch_size_train'] = 50000
NCP_trainer.learning_kwargs['shuffle_train'] = False
NCP_trainer.loss_kwargs['gamma'] = 0.001

NCP_trainer.train()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Plot unconditional density for Y [generates Figure 3.9]
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def plot_density(data,N,lim):
    # Perform KDE
    kde = gaussian_kde(data)

    x_grid, y_grid = np.meshgrid(
    np.linspace(-lim,lim, N),
        np.linspace(-lim,lim, N)
    )
    grid_coords = np.vstack([x_grid.ravel(), y_grid.ravel()])

    #density = kde(grid_coords).reshape(N,N)
    density = chi2.pdf(np.sqrt(grid_coords[0,:]**2 + grid_coords[1,:]**2),10).reshape(N,N)

    plt.figure(figsize=(5, 4))
    plt.rc('font', size=10)  # Default text size
    plt.rc('axes', titlesize=12)  # Title size
    plt.rc('lines', linewidth=1.5)  # Line thickness

    plt.imshow(
        density, origin='lower',
        extent=(-lim,lim,-lim,lim),
        cmap='hot', aspect='auto'
    )
    plt.colorbar(label='')
    plt.title('The unconditional density of Y')
    plt.xlabel('$Y_1$')
    plt.ylabel('$Y_2$')

    if _saveplots_:
        plt.savefig(save_dir + "DensityY" + ".pdf", dpi=300, bbox_inches='tight')
    else:
        plt.show()


_,Y = train.fetch(scaled=False,tensor=False)
plot_density(data = Y.T,N = 300,lim = 18)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Plot evolution for loss metrics during training [generates Figure 3.10]
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

    _saveplots_ = False
    if _saveplots_:
        plt.savefig(save_dir + 'CC_loss_metrics.pdf', dpi=300, bbox_inches='tight')
    else:
        plt.show()


plot_loss_metrics('lightning_logs/version_305/metrics.csv')

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Plot conditional mean prediction benchmark diagnostics [generates Figure 3.11]
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def calculate_conditional_mean(model,n_points,Y_sample):

    # Calculate estimated conditional mean:
    N = n_points
    u = np.linspace(0, 1, N).reshape(-1, 1)
    u = np.concatenate((u, np.random.uniform(0, 1, (u.shape[0], 9))), axis=1)
    U = train.x_scaler.transform(u)

    postprocess = 'whitening'
    y1 = model.conditional_expectation(U, Y_sample, observable=lambda y: y[:, 0].reshape(-1, 1), postprocess=postprocess)
    y2 = model.conditional_expectation(U, Y_sample, observable=lambda y: y[:, 1].reshape(-1, 1), postprocess=postprocess)
    estimated_conditional_mean = np.concatenate((y1, y2), axis=1)

    # Calculate true conditional mean under polar transform model:
    u = np.linspace(0, 1, N).reshape(-1, 1)
    U = np.concatenate((u, u), axis=1)

    true_conditional_mean = polar_transform(U, _mean_=True)
    true_conditional_mean = train.y_scaler.transform(true_conditional_mean)

    return true_conditional_mean,estimated_conditional_mean
def plot_prediction_benchmark(true_conditional_mean, estimated_conditional_mean):
    plt.figure(figsize=(4, 4))
    plt.rc('font', size=10)  # Default text size
    plt.rc('axes', titlesize=7)  # Title size
    plt.rc('lines', linewidth=1.5)  # Line thickness
    plt.scatter(true_conditional_mean[:, 0], true_conditional_mean[:, 1], facecolors='none', edgecolors='red',
                marker='o', s=80, label='True conditional mean')
    plt.scatter(estimated_conditional_mean[:, 0], estimated_conditional_mean[:, 1], marker='o', s=20,
                label='Estimated conditional mean')
    for i in range(true_conditional_mean.shape[0]):
        plt.plot([true_conditional_mean[i, 0], estimated_conditional_mean[i, 0]],
                 [true_conditional_mean[i, 1], estimated_conditional_mean[i, 1]], linewidth=0.5, color='blue')
    plt.grid(True)
    plt.legend()
    plt.show()
    # plt.savefig(save_dir + 'estimator1.pdf', dpi=300, bbox_inches='tight')


# load model
model = torch.load(NCP_trainer.paths["model_name"])

# Fetch estimated conditional mean:
_, Y = train.fetch(scaled=True, tensor=True)
true_conditional_mean, estimated_conditional_mean = calculate_conditional_mean(model,n_points= 200,Y_sample=Y)

# Plot benchmark [Figure 3.11]:
plot_prediction_benchmark(true_conditional_mean, estimated_conditional_mean)
