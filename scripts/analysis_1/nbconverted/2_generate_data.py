
# coding: utf-8

# # Generate data and calculate similarity
# 
# The goal of this notebook is to determine how much of the structure in the original dataset (single experiment) is retained after adding some number of experiments.
# 
# For this simulation experiment we wanted to capture the individual experiment structure.
# In particular, we simulated data by (1) preserving the relationship between samples within an experiment but (2) shifting the samples in space.
# 
# Criteria (1) will account for the type of experiment, such as treatment vs non-treatment.  Criteria (2) will reflect a different type of perturbation, like a different antibiotic.  
# 
# The approach is to,
# 1. Randomly sample an experiment from the Pseudomonas compendium
# 2. Embed samples from the experiment into the trained latent space
# 3. Randomly shift the samples to a new location in the latent space. This new location will be selected based on the distribution of samples in the latent space 

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import os
import sys
import glob
import pandas as pd
import numpy as np
import random

import warnings
warnings.filterwarnings(action='ignore')

sys.path.append("../")
from functions import generate_data

from numpy.random import seed
randomState = 123
seed(randomState)


# In[2]:


# User parameters
NN_architecture = 'NN_2500_30'
analysis_name = 'analysis_1'
num_simulated_experiments = 50


# In[3]:


# Input files

# base dir on repo
base_dir = os.path.abspath(os.path.join(os.getcwd(),"../..")) 

# base dir on local machine for data storage
# os.makedirs doesn't recognize `~`
local_dir = local_dir = os.path.abspath(os.path.join(os.getcwd(), "../../../..")) 

NN_dir = base_dir + "/models/" + NN_architecture

normalized_data_file = os.path.join(
    base_dir,
    "data",
    "input",
    "train_set_normalized.pcl")


# ### Load file with experiment ids

# In[4]:


experiment_ids_file = os.path.join(
    base_dir,
    "data",
    "metadata",
    "experiment_ids.txt")


# ### Generate simulated data

# In[5]:


# Generate simulated data
generate_data.simulate_compendium(experiment_ids_file, 
                                  num_simulated_experiments,
                                  normalized_data_file,
                                  NN_architecture,
                                  analysis_name
                                 )

