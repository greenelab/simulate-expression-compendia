
# coding: utf-8

# # Train VAE

# In[2]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import os
import sys
import ast
import pandas as pd
import numpy as np
import random
import warnings
warnings.filterwarnings(action='ignore')

sys.path.append("../../")
from functions import vae, utils

from numpy.random import seed
randomState = 123
seed(randomState)


# In[5]:


# Read in config variables
config_file = os.path.abspath(os.path.join(os.getcwd(),"../../configs", "config_Pa_sample.tsv"))
params = utils.read_config(config_file)


# In[2]:


# Load parameters
dataset_name = params['dataset_name']


# In[3]:


# Load arguments
base_dir = os.path.abspath(os.path.join(os.getcwd(),"../.."))

normalized_data_file = os.path.join(
    base_dir,
    dataset_name,
    "data",
    "input",
    "train_set_normalized.pcl")


# In[4]:


# Read data
normalized_data = pd.read_table(
    normalized_data_file,
    header=0,
    sep='\t',
    index_col=0).T

print(normalized_data.shape)


# In[5]:


# VAE training parameters
learning_rate = 0.001
batch_size = 100
epochs = 100
kappa = 0.01
intermediate_dim = 2500
latent_dim = 30
epsilon_std = 1.0
train_architecture = "NN_{}_{}".format(intermediate_dim, latent_dim)


# In[6]:


# Create analysis output directories
output_dirs = [os.path.join(base_dir, dataset_name, "models"),
             os.path.join(base_dir, dataset_name, "logs")
             ]

# Check if analysis output directory exist otherwise create
for each_dir in output_dirs:
    if os.path.exists(each_dir):
        print('directory already exists: {}'.format(each_dir))
    else:
        print('creating new directory: {}'.format(each_dir))
    os.makedirs(each_dir, exist_ok=True)
    

# Check if NN architecture directory exist otherwise create
for each_dir in output_dirs:
    new_dir = os.path.join(each_dir, train_architecture)
    
    if os.path.exists(new_dir):
        print('directory already exists: {}'.format(new_dir))
    else:
        print('creating new directory: {}'.format(new_dir))
    os.makedirs(new_dir, exist_ok=True)


# In[7]:


# Train (VAE)
vae.tybalt_2layer_model(learning_rate,
                        batch_size,
                        epochs, 
                        kappa, 
                        intermediate_dim,
                        latent_dim, 
                        epsilon_std,
                        normalized_data,
                        base_dir, 
                        dataset_name,
                        train_architecture)

