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

mapping_file = os.path.join(
    base_dir,
    "data",
    "metadata",
    "sample_annotations.tsv")


# In[4]:


# Output files
original_DEG_file = os.path.join(
    local_dir,
    "Data",
    "Batch_effects",
    "output",
    "analysis_1_DE_original_analysis.png")

simulated_DEG_file = os.path.join(
    local_dir,
    "Data",
    "Batch_effects",
    "output",
    "analysis_1_DE_simulated_analysis.png")


# In[5]:


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


# In[6]:


normalized_data.head(10)


# In[7]:


normalized_data.hist(column='PA0002')


# In[8]:


simulated_data.head(10)


# In[9]:


simulated_data.hist(column='PA0002')


# In[10]:


# Read in metadata
metadata = pd.read_table(
    mapping_file, 
    header=0, 
    sep='\t', 
    index_col=0)

metadata.head()


# In[11]:


map_experiment_sample = metadata[['sample_name', 'ml_data_source']]
map_experiment_sample.head()


# ## Oxygen gradient experiment
# 
# **Question:** Is the gene expression pattern/profile of the PA1683 gene consistent between the input and the simulated data?  (The magnitude of the activity may be different but the trend should be the same)

# In[12]:


# Get experiment id
experiment_id = 'E-GEOD-52445'


# In[13]:


# Get original samples associated with experiment_id
selected_mapping = map_experiment_sample.loc[experiment_id]
original_selected_sample_ids = list(selected_mapping['ml_data_source'].values)

selected_original_data = normalized_data.loc[original_selected_sample_ids]

selected_original_data.head(5)


# In[14]:


# Get first matching experiment id
match_experiment_id = ''
for experiment_name in simulated_data['experiment_id'].values:
    if experiment_name.split("_")[0] == experiment_id:
        match_experiment_id = experiment_name        


# In[15]:


# Get simulated samples associated with experiment_id
selected_simulated_data = simulated_data[simulated_data['experiment_id'] == match_experiment_id]

# Map sample ids from original data to simulated data
selected_simulated_data.index = original_selected_sample_ids

selected_simulated_data.head(5)


# In[16]:


# Plot original data
sns.clustermap(selected_original_data.T, cmap="viridis")


# In[17]:


# Plot simulated
selected_simulated_data = selected_simulated_data.drop(columns=['experiment_id'])
sns.clustermap(selected_simulated_data.T, cmap="viridis")


# ## Two different conditions

# In[18]:


# Get experiment id
#experiment_id = 'E-GEOD-43641'
experiment_id = 'E-GEOD-51409'
#experiment_id = 'E-GEOD-49759'
#experiment_id = 'E-GEOD-30967'


# In[19]:


# Get original samples associated with experiment_id
selected_mapping = map_experiment_sample.loc[experiment_id]
original_selected_sample_ids = list(selected_mapping['ml_data_source'].values)

selected_original_data = normalized_data.loc[original_selected_sample_ids]

selected_original_data.head(10)


# In[20]:


# Get first matching experiment id
match_experiment_id = ''
for experiment_name in simulated_data['experiment_id'].values:
    if experiment_name.split("_")[0] == experiment_id:
        match_experiment_id = experiment_name 


# In[21]:


# Get simulated samples associated with experiment_id
selected_simulated_data = simulated_data[simulated_data['experiment_id'] == match_experiment_id]

# Map sample ids from original data to simulated data
selected_simulated_data.index = original_selected_sample_ids

selected_simulated_data.head(5)


# In[22]:


# Plot original data
sns.clustermap(selected_original_data.T, cmap="viridis")


# In[23]:


# Plot simulated
selected_simulated_data = selected_simulated_data.drop(columns=['experiment_id'])
with sns.color_palette("viridis"):
    sns.clustermap(selected_simulated_data.T, cmap="viridis")


# ## Output selected gene expression data
# 
# We will use this to put into R script to identify differentially expressed genes (DEGs)

# In[24]:


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

# In[25]:


get_ipython().run_line_magic('load_ext', 'rpy2.ipython')
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri


# In[26]:


get_ipython().run_cell_magic('R', '', 'source("/home/alexandra/Documents/Repos/Batch_effects_simulation/scripts/functions/DE_analysis.R")\nexperiment_id = \'E-GEOD-51409\'\nfind_DEGs("metadata_deg_phosphate", experiment_id)\n#find_DEGs("metadata_deg_temp")')


# ## Visualize gene expression data using DEGs

# In[27]:


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


# In[28]:


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


# ### Select top differentially expressed genes

# In[29]:


DEG_sim_data.sort_values(by=['adj.P.Val'])
DEG_sim_data = DEG_sim_data.iloc[0:10,]

DEG_sim_data


# In[30]:


DEG_original_data.sort_values(by=['adj.P.Val'])
DEG_original_data = DEG_original_data.iloc[0:10,]

DEG_original_data


# In[31]:


# Get DEG ids
sim_gene_ids = list(DEG_sim_data.index)
original_gene_ids = list(DEG_original_data.index)


# In[32]:


# Plot original data
selected_original_DEG_data = selected_original_data[original_gene_ids]
#sns.clustermap(selected_original_DEG_data.T)
f = sns.clustermap(selected_original_DEG_data.T, cmap="viridis")
f.savefig(original_DEG_file, dpi=300)


# In[41]:


# Plot simulated
#selected_simulated_data = selected_simulated_data.drop(columns=['experiment_id'])
selected_simulated_DEG_data = selected_simulated_data[sim_gene_ids]
#sns.clustermap(selected_simulated_DEG_data.T)
import matplotlib.pyplot as plt
sns.set(style="ticks", context="talk")
plt.style.use("dark_background")
#sns.set_style("darkgrid")
sns.clustermap(selected_simulated_DEG_data.T, cmap="viridis")
#f.savefig(simulated_DEG_file, dpi=300)

