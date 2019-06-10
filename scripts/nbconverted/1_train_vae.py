
# coding: utf-8

# # Train VAE

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import os
import pandas as pd
import numpy as np
import random

from functions import vae

from numpy.random import seed
randomState = 123
seed(randomState)


# In[2]:


# Create list of base directories

base_dir = os.path.dirname(os.getcwd())

base_dirs = [os.path.join(os.path.dirname(os.getcwd()), "data"),
             os.path.join(os.path.dirname(os.getcwd()), "models"),
             os.path.join(os.path.dirname(os.getcwd()), "output"),
             os.path.join(os.path.dirname(os.getcwd()), "data", "encoded"),
             os.path.join(os.path.dirname(os.getcwd()), "output", "stats"),
             os.path.join(os.path.dirname(os.getcwd()), "output", "viz")
             ]

# Check if analysis directory exist otherwise create

for each_dir in base_dirs:

    if os.path.exists(each_dir):
        print('directory already exists: {}'.format(each_dir))
    else:
        print('creating new directory: {}'.format(each_dir))
    os.makedirs(each_dir, exist_ok=True)


# In[3]:


# Parameters 
learning_rate = 0.001
batch_size = 100
epochs = 100
kappa = 0.01
intermediate_dim = 2500
latent_dim = 300
epsilon_std = 1.0
num_PCs = latent_dim
train_architecture = "NN_{}_{}".format(intermediate_dim, latent_dim)


# In[4]:


# Create output directories

output_dirs = [os.path.join(os.path.dirname(os.getcwd()), "data", "encoded"),
             os.path.join(os.path.dirname(os.getcwd()), "models"),
             os.path.join(os.path.dirname(os.getcwd()), "output", "stats"),
             os.path.join(os.path.dirname(os.getcwd()), "output", "viz")
             ]

# Check if analysis directory exist otherwise create

for each_dir in output_dirs:
    new_dir = os.path.join(each_dir, train_architecture)
    
    if os.path.exists(new_dir):
        print('directory already exists: {}'.format(new_dir))
    else:
        print('creating new directory: {}'.format(new_dir))
    os.makedirs(new_dir, exist_ok=True)


# In[5]:


# Train nonlinear (VAE)
vae.tybalt_2layer_model(learning_rate,
                        batch_size,
                        epochs, 
                        kappa, 
                        intermediate_dim,
                        latent_dim, 
                        epsilon_std, 
                        base_dir, 
                        train_architecture 
                        )

