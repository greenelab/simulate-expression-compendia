
# coding: utf-8

# # Process data
# This notebook does the following:
# 
# 1. Selects template experiment
# 2. Downloads subset of recount2 data, including the template experiment (50 random experiments + 1 template experiment)
# 3. Train VAE on subset of recount2 data

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('load_ext', 'rpy2.ipython')
get_ipython().run_line_magic('autoreload', '2')

import os
import sys
import pandas as pd
import numpy as np
import random
import rpy2
import seaborn as sns
from sklearn import preprocessing
import pickle

sys.path.append("../")
from functions import generate_labeled_data, utils, pipeline

from numpy.random import seed
randomState = 123
seed(randomState)


# In[2]:


# Read in config variables
base_dir = os.path.abspath(os.path.join(os.getcwd(),"../"))

config_file = os.path.abspath(os.path.join(base_dir,
                                           "Rank_pathways",
                                           "init_config.tsv"))
params = utils.read_config(config_file)


# ### Select template experiment
# 
# We manually selected bioproject [SRP000762](https://trace.ncbi.nlm.nih.gov/Traces/sra/?study=SRP000762), which contains 4 A549 sample replicates - 2 were treated with 100nM Dexamethasone (DEX) and 2 were treated with 0.01% ethanol (control).
# 
# For each sample replicate there were 5-6 sequencing runs.
# 
# * [DEX treated replicate 1](https://www.ncbi.nlm.nih.gov/sra/SRX006804[accn])
# * [DEX treated replicate 2](https://www.ncbi.nlm.nih.gov/sra/SRX006806[accn])
# * [Ethanol treated replicate 1](https://www.ncbi.nlm.nih.gov/sra/SRX006805[accn])
# * [Ethanol treated replicate 2](https://www.ncbi.nlm.nih.gov/sra/SRX006807[accn])
# 
# For this analysis we will treat each run as a *sample*

# In[4]:


# Load params
local_dir = params["local_dir"]
dataset_name = params['dataset_name']
NN_architecture = params['NN_architecture']
project_id = params['project_id']


# ### Download subset of recount2 to use as a compendium
# The compendium will be composed of random experiments + the selected template experiment

# In[5]:


get_ipython().run_cell_magic('R', '', '# Select 59\n# Run one time\nif (!requireNamespace("BiocManager", quietly = TRUE))\n    install.packages("BiocManager")\nBiocManager::install("recount")')


# In[6]:


get_ipython().run_cell_magic('R', '', "library('recount')")


# In[7]:


get_ipython().run_cell_magic('R', '-i project_id -i base_dir -i local_dir', "\nsource('../functions/download_recount2_data.R')\n\nget_recount2_compendium(project_id, base_dir, local_dir)")


# ### Download expression data for selected project id

# In[12]:


get_ipython().run_cell_magic('R', '-i project_id -i local_dir', "\nsource('../functions/download_recount2_data.R')\n\nget_recount2_template_experiment(project_id, local_dir)")


# ### Normalize compendium 

# In[6]:


# Load real gene expression data
original_compendium_file = os.path.join(
    local_dir,
    "recount2_compedium_data.tsv")


# In[7]:


# Read data
original_compendium = pd.read_table(
    original_compendium_file,
    header=0,
    sep='\t',
    index_col=0)

print(original_compendium.shape)
original_compendium.head()


# In[8]:


# 0-1 normalize per gene
scaler = preprocessing.MinMaxScaler()
original_data_scaled = scaler.fit_transform(original_compendium)
original_data_scaled_df = pd.DataFrame(original_data_scaled,
                                columns=original_compendium.columns,
                                index=original_compendium.index)

original_data_scaled_df.head()


# In[9]:


# Save normalized data
normalized_data_file = os.path.join(
    local_dir,
    "normalized_recount2_compendium_data.tsv")

original_data_scaled_df.to_csv(
    normalized_data_file, float_format='%.3f', sep='\t')

# Save scaler transform
scaler_file = os.path.join(
    local_dir,
    "scaler_transform.pickle")

outfile = open(scaler_file,'wb')
pickle.dump(scaler,outfile)
outfile.close()


# ### Train VAE 

# In[17]:


# Setup directories
# Create VAE directories
output_dirs = [os.path.join(base_dir, dataset_name, "models"),
               os.path.join(base_dir, dataset_name, "logs")]

# Check if analysis output directory exist otherwise create
for each_dir in output_dirs:
    if os.path.exists(each_dir) == False:
        print('creating new directory: {}'.format(each_dir))
        os.makedirs(each_dir, exist_ok=True)

# Check if NN architecture directory exist otherwise create
for each_dir in output_dirs:
    new_dir = os.path.join(each_dir, NN_architecture)
    if os.path.exists(new_dir) == False:
        print('creating new directory: {}'.format(new_dir))
        os.makedirs(new_dir, exist_ok=True)


# In[14]:


# Train VAE on new compendium data
# Write out model to rank_pathways directory
pipeline.train_vae(config_file,
                   normalized_data_file)

