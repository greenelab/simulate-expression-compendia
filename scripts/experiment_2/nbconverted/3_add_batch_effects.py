#!/usr/bin/env python
# coding: utf-8

# # Add batch effects
# 
# Say we are interested in identifying genes that differentiate between disease vs normal states.  However our dataset includes samples from different tissues or time points and there are variations in gene expression that are due to these other conditions and do not have to do with disease state.  These non-relevant variations in the data are called *batch effects*.  
# 
# We want to model these batch effects.  To do this we will:
# 1. Partition our simulated data into n batches
# 2. For each partition we will randomly shift the expression data.  We randomly generate a binary vector of length=number of genes (*offset vector*).  This vector will serve as the direction that we will shift to.  Then we also have a random scalar that will tell us how big of a step to take in our random direction (*stretch factor*).  We shift our partitioned data by: batch effect partition = partitioned data + stretch factor * offset vector
# 3. Repeat this for each partition
# 4. Append all batch effect partitions together
# 

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import os
import ast
import pandas as pd
import numpy as np
import random
import glob
import umap
import pickle
import seaborn as sns
import warnings
warnings.filterwarnings(action='ignore')

from sklearn.decomposition import PCA
from numpy.random import seed
randomState = 123
seed(randomState)


# In[2]:


# Load config file
config_file = "config_exp_2.txt"

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


# In[3]:


# Parameters
analysis_name = d["analysis_name"]
NN_architecture = d["NN_architecture"]
num_PCs = d["num_PCs"]
num_batches = d["num_batches"]


# In[4]:


# Create directories
base_dir = os.path.abspath(os.path.join(os.getcwd(),"../.."))

new_dir = os.path.join(
    base_dir,
    "data",
    "batch_simulated")

analysis_dir = os.path.join(new_dir, analysis_name)

if os.path.exists(analysis_dir):
    print('directory already exists: {}'.format(analysis_dir))
else:
    print('creating new directory: {}'.format(analysis_dir))
os.makedirs(analysis_dir, exist_ok=True)


# In[5]:


# Load arguments
simulated_data_file = os.path.join(
    base_dir,
    "data",
    "simulated",
    analysis_name,
    "simulated_data.txt.xz")

umap_model_file = umap_model_file = os.path.join(
    base_dir,
    "models",  
    NN_architecture,
    "umap_model.pkl")


# In[6]:


# Read in UMAP model
infile = open(umap_model_file, 'rb')
umap_model = pickle.load(infile)
infile.close()


# In[7]:


# Read in data
simulated_data = pd.read_table(
    simulated_data_file,
    header=0, 
    index_col=0,
    compression='xz',
    sep='\t')

simulated_data.head(10)


# In[8]:


get_ipython().run_cell_magic('time', '', '# Add batch effects\nnum_simulated_samples = simulated_data.shape[0]\nnum_genes = simulated_data.shape[1]\n\n# Create an array of the simulated data indices\nsimulated_ind = np.array(simulated_data.index)\n\nfor i in num_batches:\n    print(\'Creating simulated data with {} batches..\'.format(i))\n    \n    batch_file = os.path.join(\n            base_dir,\n            "data",\n            "batch_simulated",\n            analysis_name,\n            "Batch_"+str(i)+".txt.xz")\n    \n    num_samples_per_batch = int(num_simulated_samples/i)\n    \n    if i == 1:        \n        simulated_data.to_csv(batch_file, sep=\'\\t\', compression=\'xz\')\n        \n    else:  \n        batch_data = simulated_data.copy()\n        \n        # Shuffle indices\n        np.random.shuffle(simulated_ind)\n        \n        for j in range(i):\n            #print(j)\n            \n            # Partition indices to batch\n            partition = np.array_split(simulated_ind, i)\n            \n            #print("before")\n            #print(batch_data.loc[partition[j].tolist()].head())\n            \n            #print("indices to change: {}".format(partition))\n            \n            # Scalar to shift gene expressiond data\n            stretch_factor = np.random.normal(0.0, 0.2, [1,num_genes])\n            \n            #print(stretch_factor)\n            \n            # Tile stretch_factor to be able to add to batches\n            num_samples_per_batch = len(partition[j])\n            stretch_factor_tile = pd.DataFrame(\n                pd.np.tile(\n                    stretch_factor,\n                    (num_samples_per_batch, 1)),\n                index=batch_data.loc[partition[j].tolist()].index,\n                columns=batch_data.loc[partition[j].tolist()].columns)\n            \n            #print(stretch_factor_tile.head())\n            \n            # Add batch effects\n            batch_data.loc[partition[j].tolist()] = batch_data.loc[partition[j].tolist()] + stretch_factor_tile\n            \n            #print("after")\n            #print(batch_data.loc[partition[j].tolist()].head())\n\n\n        # Should we re-normalize from 0-1 range?\n        #from sklearn import preprocessing\n        #batch_data = preprocessing.MinMaxScaler().fit_transform(batch_data)\n        #batch_data_df = pd.DataFrame(batch_data,\n        #                        columns=batch_data.columns,\n        #                        index=batch_data.index)\n            \n        #print(batch_data)\n            \n        # Save\n        batch_data.to_csv(batch_file, sep=\'\\t\', compression=\'xz\')')

