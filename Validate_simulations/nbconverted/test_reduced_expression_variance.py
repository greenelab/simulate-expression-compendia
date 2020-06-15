
# coding: utf-8

# # Test reduced variance of gene expression data
# 
# **Motivation**: When we plotted a volcano plot of the E-GEOD-51409 array experiment using the [actual data](volcano_original_data_E-GEOD-51409_example_adjp.png) and the [experiment-level simulated data](volcano_simulated_data_E-GEOD-51409_example_adjp.png), we found that the simulated data had reduced variance based on the squished log fold chance values.
# 
# **Question:** What is causing the reduced variance in the simulated data? Is this reduced variance 1) a property of the variational autoencoder (VAE) algorithm or 2) the result of the latent space shifting (see [simulate_compendium module](../functions/generate_data_parallel.py)?
# 
# **Approach:**
# This notebook aims to answer this question by performing 3 short experiments, each building off of the next. 
# 1. In the first experiment, we test the effect of applying the VAE. 
# 2. In the second experiment, we test the effect of sampling from the VAE latent space to simulate gene expression data (i.e. [sample-level-simulation](../Pseudomonas/Pseudomonas_sample_lvl_sim.ipynb) approach). 
# 3. In the third experiment, we test the effect of the latent space shifting (i.e. [experiment-level simulation](../Pseudomonas/Pseudomonas_experiment_lvl_sim.ipynb) approach).
# 
# **Conclusion:** The VAE is minimally contributes the the reduced variance. Most of the reduced variance is due to the sampling of the latent space due to the Normal constraint of the latent space.

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import os
import ast
import sys
import glob
import pandas as pd
import numpy as np
import random
from keras.models import load_model

from ponyo import utils

import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

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
dataset_name = params['dataset_name']
NN_architecture = params['NN_architecture']


# ## Experiment 1
# ### Testing the effect of the VAE
# 
# In this experiment we want to test the effect of the VAE on the gene expression variance. 
# 
# To do this we will compare the distribution of the gene variance in the [original dataset](../Pseudomonas/data/input/train_set_normalized.pcl), which does not use the VAE, versus the original data after it has been encoded and decoded by the VAE. The only variable we are varying is the application of the VAE.

# In[4]:


# Load datasets from the actual data and the simulate data
no_vae_file = os.path.join(
    base_dir,
    "Pseudomonas",
    "data",
    "input",
    "train_set_normalized.pcl")

# Load VAE encoder and decoder models
NN_dir = os.path.join(
    base_dir, 
    dataset_name,
    "models",
    NN_architecture)
model_encoder_file = glob.glob(os.path.join(
    NN_dir,
    "*_encoder_model.h5"))[0]

weights_encoder_file = glob.glob(os.path.join(
    NN_dir,
    "*_encoder_weights.h5"))[0]

model_decoder_file = glob.glob(os.path.join(
    NN_dir,
    "*_decoder_model.h5"))[0]

weights_decoder_file = glob.glob(os.path.join(
    NN_dir,
    "*_decoder_weights.h5"))[0]

loaded_model = load_model(model_encoder_file)
loaded_decode_model = load_model(model_decoder_file)

loaded_model.load_weights(weights_encoder_file)
loaded_decode_model.load_weights(weights_decoder_file)


# In[5]:


# Read in datasets
no_vae_data = pd.read_table(
    no_vae_file,
    header=0,
    index_col=0,
    sep='\t').T

print(no_vae_data.shape)
no_vae_data.head()


# In[6]:


# Pass original data through VAE
# Encode selected experiment into latent space
data_encoded = loaded_model.predict_on_batch(no_vae_data)
data_encoded_df = pd.DataFrame(
    data_encoded, 
    index=no_vae_data.index)

# Decode simulated data into raw gene space
data_decoded = loaded_decode_model.predict_on_batch(data_encoded_df)

vae_data = pd.DataFrame(data_decoded,
                        index=data_encoded_df.index,
                        columns=no_vae_data.columns)

print(vae_data.shape)
vae_data.head()


# In[7]:


# Get variance per gene
var_no_vae = no_vae_data.var(axis=0)
var_vae= vae_data.var(axis=0)


# In[8]:


df_vae = pd.DataFrame(list(zip(var_no_vae, var_vae)), 
               columns =['original', 'original after VAE']) 
df_vae.head()


# In[9]:


# Plot distribution of variances using original data and original data passed through the VAE
boxplot = df_vae.boxplot(column=['original', 'original after VAE'])
_ = boxplot.set_title("Distribution of per-gene expression variances")
_ = boxplot.set_ylabel("variances per gene")


# **Observations:** The VAE model (encoder + decoder) does not appear to have much of an effect on the variance. This is expected, given that the model was trained to reconstruct the input data

# ## Experiment 2
# ### Testing the effect of sampling from the latent space to simulate data
# 
# In this experiment we want to test the effect of sampling from the VAE latent space on the gene expression variance. 
# 
# To do this we will compare the distribution of the gene variance in the [original dataset](../Pseudomonas/data/input/train_set_normalized.pcl) versus a simulated dataset generated using the [sample-level simulation](../Pseudomonas/Pseudomonas_sample_lvl_sim.ipynb), which uses the learned latent space of the VAE to simulate new data. Building off of the results from experiment 1, we add sampling from the latent space as a factor in this experiment.

# In[10]:


# Load datasets from the actual data and the sample-simulated data
no_sampling_file = os.path.join(
    base_dir,
    "Pseudomonas",
    "data",
    "input",
    "train_set_normalized.pcl")

vae_sampling_file = os.path.join(
    local_dir,
    "experiment_simulated",
    "Pseudomonas_sample_lvl_sim",
    "Experiment_1_0.txt.xz")


# In[11]:


# Read in datasets
no_sampling_data = pd.read_table(
    no_sampling_file,
    header=0,
    index_col=0,
    sep='\t').T

vae_sampling_data = pd.read_table(
    vae_sampling_file,
    header=0,
    index_col=0,
    sep='\t')

print(no_sampling_data.shape)
no_sampling_data.head()


# In[12]:


print(vae_sampling_data.shape)
vae_sampling_data.head()


# In[13]:


# Get variance per gene
var_no_sampling = no_sampling_data.var(axis=0)
var_vae_sampling= vae_sampling_data.var(axis=0)


# In[14]:


df_sampling = pd.DataFrame(list(zip(var_no_sampling, var_vae_sampling)), 
               columns =['original', 'sample simulated']) 
df_sampling.head()


# In[15]:


# Plot distribution of variances using original and sample-simulated data
boxplot = df_sampling.boxplot(column=['original', 'sample simulated'])
_ = boxplot.set_title("Distribution of per-gene expression variances")
_ = boxplot.set_ylabel("variances per gene")


# **Observations**:
# Based on the results it looks like there is some shrinkage of the variance using the VAE latent space to sample from. This is expected given the assumption that the VAE is making for latent space features to draw from a standard Normal distribution. Without this constraint, there are sparse regions in the latent space (i.e. the space is not continuous) that make generating relalistic 
# data from these regions difficult because there is no information about this space. Therefore, this constraint was added ontop of the generic autoencoder (AE) in order to reduce the variance in the latent space to ensure a continuous latent space. Thus we would expect the VAE to squish the variance of the original data.
# 
# 
# Some references:
# - https://arxiv.org/abs/1312.6114
# - https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf

# ## Experiment 3
# ### Testing the effect of the latent space shift to simulate data
# 
# In this experiment we want to test the effect of the linear shift in the latent space on the gene expression variance. 
# 
# To do this we will compare the distribution of the gene variance in the simulated dataset generated using the [sample-level simulation](../Pseudomonas/Pseudomonas_sample_lvl_sim.ipynb), which does *not* perform a linear shift, versus the simulated dataset generated using the [experiment-level simulation](../Pseudomonas/Pseudomonas_experiment_lvl_sim.ipynb), which does perform a linear shift. Both simulations use the VAE, so this factor is held constant and the variable we are testing is the latent space shift.
# 

# In[16]:


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


# In[17]:


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


# In[18]:


print(shift_data.shape)
shift_data.head()


# In[19]:


# Get variance per gene
var_no_shift = no_shift_data.var(axis=0)
var_shift = shift_data.var(axis=0)


# In[20]:


df_shift = pd.DataFrame(list(zip(var_no_shift, var_shift)), 
               columns =['sample simulated(not shifted)', 'experiment simulated (shifted)']) 
df_shift.head()


# In[21]:


# Plot distribution of variances using no shifted data and shifted data
boxplot = df_shift.boxplot(column=['sample simulated(not shifted)', 
                                   'experiment simulated (shifted)'])
_ = boxplot.set_title("Distribution of per-gene expression variances")
_ = boxplot.set_ylabel("variances per gene")


# **Observations:** We can see that compared to the not shifted compendium, the shifted compendium has a slightly larger variance. This makes sense given that we are shifting our samples in the latent space. The samples were shifted randomly to a new location in the latent space. Theoretically, we could get a smaller variance using the shifted approach *if* all the shifts happen to compress the samples together, however the likelihood of this happening is very rare.

# ## Summary
# 
# Plotting the distribution of variances per-gene for all cases (original data, sampling from VAE space, sampling and shifting in VAE space), we can see that the largest reduction in variance is due to the sampling of the VAE space.

# In[22]:


df_shift = pd.DataFrame(list(zip(var_no_shift, var_shift, var_no_sampling)), 
               columns =['sample simulated(not shifted)', 'experiment simulated (shifted)', 'original']) 
df_shift.head()


# In[23]:


# Plot distribution of variances using no shifted data and shifted data
boxplot = df_shift.boxplot(column=['sample simulated(not shifted)', 
                                   'experiment simulated (shifted)', 
                                   'original'],
                          rot=45)
_ = boxplot.set_title("Distribution of per-gene expression variances")
_ = boxplot.set_ylabel("variances per gene")

