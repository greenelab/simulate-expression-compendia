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


get_ipython().run_cell_magic('time', '', '# Add batch effects\nnum_simulated_samples = simulated_data.shape[0]\nnum_genes = simulated_data.shape[1]\nsubset_genes_to_change = np.random.RandomState(randomState).choice([0, 1], size=(num_genes), p=[3./4, 1./4])\n    \nfor i in num_batches:\n    print(\'Creating simulated data with {} batches..\'.format(i))\n    \n    batch_file = os.path.join(\n            base_dir,\n            "data",\n            "batch_simulated",\n            analysis_name,\n            "Batch_"+str(i)+".txt.xz")\n    \n    num_samples_per_batch = int(num_simulated_samples/i)\n    \n    if i == 1:        \n        simulated_data.to_csv(batch_file, sep=\'\\t\', compression=\'xz\')\n        \n    else:  \n        batch_data_df = pd.DataFrame()\n        \n        simulated_data_draw = simulated_data\n        for j in range(i):            \n            # Randomly select samples\n            batch_df = simulated_data_draw.sample(n=num_samples_per_batch, frac=None, replace=False)\n            batch_df.columns = batch_df.columns.astype(str)\n            \n            # Update df to remove selected samples\n            sampled_ids = list(batch_df.index)\n            simulated_data_draw = simulated_data_draw.drop(sampled_ids)\n\n            # Add batch effect\n            \n            # Option 1: Add small amount to subset of genes\n            #stretch_factor = np.random.uniform(0,0.5)\n            #stretch_factor = 0.0\n            #subset_genes_to_change_tile = pd.DataFrame(\n            #    pd.np.tile(\n            #        subset_genes_to_change,\n            #        (num_samples_per_batch, 1)),\n            #    index=batch_df.index,\n            #    columns=simulated_data.columns)\n\n            #offset_vector = pd.DataFrame(subset_genes_to_change_tile*stretch_factor)\n            #offset_vector.columns = offset_vector.columns.astype(str)\n            \n            #batch_df = batch_df + offset_vector\n            \n            # Option 2: Multiply all samples by small scale factor\n            stretch_factor = np.random.uniform(1.0,1.5)\n            batch_df = batch_df*stretch_factor\n\n            # if any exceed 1 then set to 1 since gene expression is normalized\n            batch_df[batch_df>=1.0] = 1.0\n\n            # Append batched together\n            batch_data_df = batch_data_df.append(batch_df)\n\n            # Select a new direction (i.e. a new subset of genes to change)\n            np.random.shuffle(subset_genes_to_change)\n            \n        # Save\n        batch_data_df.to_csv(batch_file, sep=\'\\t\', compression=\'xz\')')


# ## Plot batch data using UMAP

# In[9]:


"""
# Plot generated data 

for i in num_batches:
    batch_data_file = os.path.join(
        base_dir,
        "data",
        "batch_simulated",
        analysis_name,
        "Batch_"+str(i)+".txt")
    
    batch_data = pd.read_table(
        batch_data_file,
        header=0,
        sep='\t',
        index_col=0)
    
    # UMAP embedding of decoded batch data
    batch_data_UMAPencoded = umap_model.transform(batch_data)
    batch_data_UMAPencoded_df = pd.DataFrame(data=batch_data_UMAPencoded,
                                             index=batch_data.index,
                                             columns=['1','2'])
    
        
    g = ggplot(aes(x='1',y='2'), data=batch_data_UMAPencoded_df) + \
                geom_point(alpha=0.5) + \
                scale_color_brewer(type='qual', palette='Set2') + \
                ggtitle("{} Batches".format(i))
    
    print(g)"""


# ## Plot batch data using PCA

# In[10]:


"""
# Plot generated data 

for i in num_batches:
    batch_data_file = os.path.join(
        base_dir,
        "data",
        "batch_simulated",
        analysis_name,
        "Batch_"+str(i)+".txt")
    
    batch_data = pd.read_table(
        batch_data_file,
        header=0,
        sep='\t',
        index_col=0)
    
    # PCA projection    
    pca = PCA(n_components=num_PCs)
    batch_data_PCAencoded = pca.fit_transform(batch_data)
    
    # Encode data using PCA model    
    batch_data_PCAencoded_df = pd.DataFrame(batch_data_PCAencoded,
                                         index=batch_data.index
                                         )
    
    g = sns.pairplot(batch_data_PCAencoded_df)
    g.fig.suptitle("Batch {}".format(i))
       
    # Select pairwise PC's to plot
    pc1 = 0
    pc2 = 2
    
    # Encode data using PCA model    
    batch_data_PCAencoded_df = pd.DataFrame(batch_data_PCAencoded[:,[pc1,pc2]],
                                         index=batch_data.index,
                                         columns=['PC {}'.format(pc1), 'PC {}'.format(pc2)])
    
    g = ggplot(aes(x='PC {}'.format(pc1),y='PC {}'.format(pc2)), data=batch_data_PCAencoded_df)  + \
                geom_point(alpha=0.5) + \
                scale_color_brewer(type='qual', palette='Set2') + \
                ggtitle("{} Batches".format(i))
    print(g)

"""

