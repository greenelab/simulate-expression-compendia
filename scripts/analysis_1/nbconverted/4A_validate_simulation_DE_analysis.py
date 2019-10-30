#!/usr/bin/env python
# coding: utf-8

# # Visualize biological trends in the data
# 
# This notebook aims to compare the biological trends in the simulated data and the original data in order to validate using this VAE to generate gene expression data.
# 
# In this simulation experiment we are preserving the experiment type but not the actual experiment so the relationship between samples within an experiment are preserved but the genes that are expressed will be different

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import os
import ast
import pandas as pd
import numpy as np
import seaborn as sns
import random
import glob
from sklearn import preprocessing

import warnings
warnings.filterwarnings(action='ignore')

from numpy.random import seed
randomState = 123
seed(randomState)


# In[2]:


# User parameters
NN_architecture = 'NN_2500_30'
analysis_name = 'analysis_1'


# In[3]:


# Load data

# base dir on repo
base_dir = os.path.abspath(os.path.join(os.getcwd(),"../.."))          

# base dir on local machine for data storage
# os.makedirs doesn't recognize `~`
local_dir = os.path.abspath(os.path.join(os.getcwd(), "../../../.."))  
    
latent_dim = NN_architecture.split('_')[-1]

NN_dir = base_dir + "/models/" + NN_architecture

normalized_data_file = os.path.join(
    base_dir,
    "data",
    "input",
    "train_set_normalized.pcl")

simulated_data_file = os.path.join(
    local_dir,
    "Data",
    "Batch_effects",
    "simulated",
    analysis_name,
    "simulated_data.txt.xz")


# In[4]:


# Read data
normalized_data = pd.read_table(
    normalized_data_file,
    header=0,
    sep='\t',
    index_col=0).T

simulated_data = pd.read_table(
    simulated_data_file,
    header=0,
    sep='\t',
    index_col=0)

print(normalized_data.shape)
print(simulated_data.shape)


# In[5]:


normalized_data.head(10)


# In[6]:


normalized_data.hist(column='PA0002')


# In[7]:


simulated_data.head(10)


# In[8]:


simulated_data.hist(column='PA0002')


# ## Oxygen gradient experiment
# 
# **Question:** Is the gene expression pattern/profile of the PA1683 gene consistent between the input and the simulated data?  (The magnitude of the activity may be different but the trend should be the same)

# In[9]:


# Get experiment id
experiment_id = 'E-GEOD-52445'


# In[10]:


# Get simulated samples associated with experiment_id
selected_simulated_data = simulated_data[simulated_data['experiment_id'] == experiment_id]

# Get sample ids associated with experiment_id
selected_sample_ids = list(selected_simulated_data.index)

selected_simulated_data.head(5)


# In[11]:


# Get original samples associated with experiment_id
selected_original_data = normalized_data.loc[selected_sample_ids]

selected_original_data.head(5)


# In[12]:


# Plot original data
sns.clustermap(selected_original_data.T)


# In[13]:


# Plot simulated
selected_simulated_data = selected_simulated_data.drop(columns=['experiment_id'])
sns.clustermap(selected_simulated_data.T)


# ## Two different conditions

# In[14]:


# Get experiment id
#experiment_id = 'E-GEOD-43641'
#experiment_id = 'E-GEOD-51409'
#experiment_id = 'E-GEOD-49759'
experiment_id = 'E-GEOD-30967'


# In[15]:


# Get simulated samples associated with experiment_id
selected_simulated_data = simulated_data[simulated_data['experiment_id'] == experiment_id]

# Get sample ids associated with experiment_id
selected_sample_ids = list(selected_simulated_data.index)

selected_simulated_data.head(5)


# In[16]:


# Get original samples associated with experiment_id
selected_original_data = normalized_data.loc[selected_sample_ids]

selected_original_data.head(10)


# In[17]:


# Plot original data
sns.clustermap(selected_original_data.T)


# In[18]:


# Plot simulated
selected_simulated_data = selected_simulated_data.drop(columns=['experiment_id'])
sns.clustermap(selected_simulated_data.T)


# ## Output selected gene expression data
# 
# We will use this to put into R script to identify differentially expressed genes (DEGs)

# In[19]:


selected_simulated_data_file = os.path.join(
    local_dir,
    "Data",
    "Batch_effects",
    "simulated",
    analysis_name,
    "selected_simulated_data.txt")

selected_original_data_file = os.path.join(
    local_dir,
    "Data",
    "Batch_effects",
    "simulated",
    analysis_name,
    "selected_original_data.txt")

selected_simulated_data.to_csv(
        selected_simulated_data_file, float_format='%.3f', sep='\t')

selected_original_data.to_csv(
        selected_original_data_file, float_format='%.3f', sep='\t')


# ## Find differentially expressed genes

# In[20]:


get_ipython().run_line_magic('load_ext', 'rpy2.ipython')
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri


# In[22]:


get_ipython().run_cell_magic('R', '', 'source("/home/alexandra/Documents/Repos/Batch_effects_simulation/scripts/functions/DE_analysis.R")\nexperiment_id = \'E-GEOD-30967\'\nfind_DEGs("metadata_deg_phosphate", experiment_id)\n#find_DEGs("metadata_deg_temp")')


# ## Visualize gene expression data using DEGs

# In[23]:


# Import list of DEGs
DEG_sim_file = os.path.join(
    local_dir,
    "Data",
    "Batch_effects",
    "simulated",
    analysis_name,
    "sign_DEG_sim_"+experiment_id+".txt")

DEG_original_file = os.path.join(
    local_dir,
    "Data",
    "Batch_effects",
    "simulated",
    analysis_name,
    "sign_DEG_original_"+experiment_id+".txt")


# In[24]:


# Read data
DEG_sim_data = pd.read_table(
    DEG_sim_file,
    header=0,
    sep='\t',
    index_col=0)

DEG_original_data = pd.read_table(
    DEG_original_file,
    header=0,
    sep='\t',
    index_col=0)

DEG_sim_data.head()


# In[25]:


# Get DEG ids
sim_gene_ids = list(DEG_sim_data.index)
original_gene_ids = list(DEG_original_data.index)


# In[26]:


# Plot original data
selected_original_DEG_data = selected_original_data[original_gene_ids]
sns.clustermap(selected_original_DEG_data.T)


# In[27]:


# Plot simulated
#selected_simulated_data = selected_simulated_data.drop(columns=['experiment_id'])
selected_simulated_DEG_data = selected_simulated_data[sim_gene_ids]
sns.clustermap(selected_simulated_DEG_data.T)

