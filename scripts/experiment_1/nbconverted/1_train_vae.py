#!/usr/bin/env python
# coding: utf-8

# # Train VAE

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import os
import ast
import pandas as pd
import numpy as np
import random
import warnings
warnings.filterwarnings(action='ignore')

from functions import vae

from numpy.random import seed
randomState = 123
seed(randomState)


# In[2]:


# Create list of base directories

base_dir = os.path.abspath(os.path.join(os.getcwd(),"../.."))

base_dirs = [os.path.join(base_dir, "data"),
             os.path.join(base_dir, "models"),
             os.path.join(base_dir, "output"),
             os.path.join(base_dir, "data", "encoded"),
             os.path.join(base_dir, "output", "stats"),
             os.path.join(base_dir, "output", "viz")
             ]

# Check if analysis directory exist otherwise create

for each_dir in base_dirs:

    if os.path.exists(each_dir):
        print('directory already exists: {}'.format(each_dir))
    else:
        print('creating new directory: {}'.format(each_dir))
    os.makedirs(each_dir, exist_ok=True)


# In[10]:


# Load config params
config_file = "config_exp_1.txt"

d = {}
float_params = ["learning_rate", "kappa", "epsilon_std"]
str_params = ["analysis_name", "NN_architecture"]
lst_params = ["num_batches"]
with open(config_file) as f:
    for line in f:
        (name, val) = line.split()
        if name in float_params:
            d[name] = float(val)
        elif name in str_params:
            d[name] = str(val)
        elif name in lst_params:
            d[name] = ast.literal_eval(val)
        else:
            d[name] = int(val)


# In[4]:


# Parameters
num_dims = d["num_dims"]


# In[5]:


# Load arguments
normalized_data_file = os.path.join(
    os.path.abspath(os.path.join(os.getcwd(),"../..")),
    "data",
    "input",
    "train_set_normalized.pcl")


# In[6]:


# Read data
normalized_data = pd.read_table(
    normalized_data_file,
    header=0,
    sep='\t',
    index_col=0).T

print(normalized_data.shape)


# In[7]:


# Parameters 
learning_rate = d['learning_rate']
batch_size = d['batch_size']
epochs = d['epochs']
kappa = d['kappa']
intermediate_dim = d['intermediate_dim']
latent_dim = d['latent_dim']
epsilon_std = d['epsilon_std']
num_PCs = latent_dim
train_architecture = "NN_{}_{}".format(intermediate_dim, latent_dim)


# In[8]:


# Create output directories

output_dirs = [os.path.join(base_dir, "data", "encoded"),
             os.path.join(base_dir, "models"),
             os.path.join(base_dir, "output", "stats"),
             os.path.join(base_dir, "output", "viz")
             ]

# Check if analysis directory exist otherwise create

for each_dir in output_dirs:
    new_dir = os.path.join(each_dir, train_architecture)
    
    if os.path.exists(new_dir):
        print('directory already exists: {}'.format(new_dir))
    else:
        print('creating new directory: {}'.format(new_dir))
    os.makedirs(new_dir, exist_ok=True)


# In[9]:


# Train nonlinear (VAE)
vae.tybalt_2layer_model(learning_rate,
                        batch_size,
                        epochs, 
                        kappa, 
                        intermediate_dim,
                        latent_dim, 
                        epsilon_std,
                        normalized_data,
                        base_dir, 
                        train_architecture 
                        )

