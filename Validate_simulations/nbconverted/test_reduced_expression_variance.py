
# coding: utf-8

# # Test reduced variance of gene expression data
# 
# **Motivation**: When we plotted a volcano plot of the E-GEOD-51409 array experiment using the [actual data](volcano_original_data_E-GEOD-51409_example_adjp.png) and the [experiment-level simulated data](volcano_simulated_data_E-GEOD-51409_example_adjp.png), we found that the simulated data had reduced variance based on the squished log fold chance values.
# 
# **Question:** What is causing the reduced variance in the simulated data? Is this reduced variance 1) the result of the latent space shifting (see [simulate_compendium module](../functions/generate_data_parallel.py) or 2) a property of the variational autoencoder (VAE) algorithm?
# 
# This notebook aims to answer this question by performing 2 short experiments. In the first experiment, we test the effect of the latent space shifting and hold the VAE constant. In the second experiment, we test the effect of the VAE and hold the shifting constant.

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import os
import ast
import sys
import pandas as pd
import numpy as np
import random

sys.path.append("../")
from functions import utils

import warnings
warnings.filterwarnings(action='ignore')

from numpy.random import seed
randomState = 123
seed(randomState)


# In[2]:


# Read in config variables
# Pick one of the Pseudomonas config files
# Doesn't matter if sample or experiment level for this notebook
base_dir = os.path.abspath(os.path.join(os.getcwd(),"../"))
config_file = os.path.abspath(os.path.join(base_dir,
                                           "configs", 
                                           "config_Pa_sample_limma.tsv"))
params = utils.read_config(config_file)


# In[3]:


# Load parameters
local_dir = params["local_dir"]


# ## Experiment 1
# 
# In this experiment we want to test the effect of the linear shift on the gene expression variance. 
# 
# To do this we will compare the distribution of the gene variance in the simulated dataset generated using the [sample-level simulation](../Pseudomonas/Pseudomonas_sample_lvl_sim.ipynb), which does *not* perform a linear shift, versus the simulated dataset generated using the [experiment-level simulation](../Pseudomonas/Pseudomonas_experiment_lvl_sim.ipynb), which does perform a linear shift. Both simulations use the VAE, so this factor is held constant and the variable we are testing is the latent space shift.
# 

# In[4]:


# Load datasets created from the two different simulations
no_shift_file = os.path.join(
    local_dir,
    "experiment_simulated",
    "Pseudomonas_sample_lvl_sim",
    "Experiment_1_0.txt.xz")

shift_file = os.path.join(
    local_dir,
    "partition_simulated",
    "Pseudomonas_experiment_lvl_sim",
    "Partition_1_0.txt.xz")


# In[5]:


# Read in datasets
no_shift_data = pd.read_table(
    no_shift_file,
    header=0,
    index_col=0,
    sep='\t')

shift_data = pd.read_table(
    shift_file,
    header=0,
    index_col=0,
    sep='\t')

print(no_shift_data.shape)
no_shift_data.head()


# In[6]:


print(shift_data.shape)
shift_data.head()


# In[7]:


# Get variance per gene
var_no_shift = no_shift_data.var(axis=0)
var_shift = shift_data.var(axis=0)


# In[9]:


df = pd.DataFrame(list(zip(var_no_shift, var_shift)), 
               columns =['not shifted', 'shifted']) 
df.head()


# In[10]:


# Plot distribution of variances using no shifted data and shifted data
boxplot = df.boxplot(column=['not shifted', 'shifted'])


# **Observations:** We can see that compared to the not shifted compendium, the shifted compendium has a slightly larger variance. This makes sense given that we are shifting our samples in the latent space. The samples were shifted randomly to a new location in the latent space. Theoretically, we could get a smaller variance using the shifted approach *if* all the shifts happen to compress the samples together, however the likelihood of this happening is very rare.

# ## Experiment 2
# 
# In this experiment we want to test the effect of the VAE on the gene expression variance. 
# 
# To do this we will compare the distribution of the gene variance in the [original dataset](../Pseudomonas/data/input/train_set_normalized.pcl), which does not use the VAE, versus simulated dataset generated using the [sample-level simulation](../Pseudomonas/Pseudomonas_sample_lvl_sim.ipynb), which uses the learned latent space of the VAE to simulate new data. Neither dataset uses the shifting, so this factor is held constant and the variable we are testing is the application of the VAE.

# In[11]:


# Load datasets from the actual data and the simulate data
no_vae_file = os.path.join(
    base_dir,
    "Pseudomonas",
    "data",
    "input",
    "train_set_normalized.pcl")

vae_file = os.path.join(
    local_dir,
    "partition_simulated",
    "Pseudomonas_experiment_lvl_sim",
    "Partition_1_0.txt.xz")


# In[12]:


# Read in datasets
no_vae_data = pd.read_table(
    no_vae_file,
    header=0,
    index_col=0,
    sep='\t').T

vae_data = shift_data

print(no_vae_data.shape)
no_vae_data.head()


# In[13]:


print(vae_data.shape)
vae_data.head()


# In[14]:


# Get variance per gene
var_no_vae = no_vae_data.var(axis=0)
var_vae= vae_data.var(axis=0)


# In[15]:


df_vae = pd.DataFrame(list(zip(var_no_vae, var_vae)), 
               columns =['no vae', 'vae']) 
df_vae.head()


# In[16]:


# Plot distribution of variances using no shifted data and shifted data
boxplot = df_vae.boxplot(column=['no vae', 'vae'])


# **Conclusions**:
# Based on the results it looks like there is some shrinkage of the variance using the VAE. This is expected given the assumption that the VAE is making for latent space features to draw from a standard Normal distribution. Without this constraint, there are sparse regions in the latent space (i.e. the space is not continuous) that make generating relalistic 
# data from these regions difficult because there is no information about this space. Therefore, this constraint was added ontop of the generic autoencoder (AE) in order to reduce the variance in the latent space to ensure a continuous latent space. Thus we would expect the VAE to squish the variance of the original data.
# 
# 
# Some references:
# - https://arxiv.org/abs/1312.6114
# - https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf
