#!/usr/bin/env python
# coding: utf-8

# # Similarity analysis
# 
# We want to determine if the different batch simulated data is able to capture the biological signal that is present in the original data:  How much of the real input data is captured in the simulated batch data?
# 
# In other words, we want to compare the representation of the real input data and the simulated batch data.  We will use **SVCCA** to compare these two representations.
# 
# Here, we apply Singular Vector Canonical Correlation Analysis [Raghu et al. 2017](https://arxiv.org/pdf/1706.05806.pdf) [(github)](https://github.com/google/svcca) to the UMAP and PCA representations of our batch 1 simulated dataset vs batch n simulated datasets.  The output of the SVCCA analysis is the SVCCA mean similarity score. This single number can be interpreted as a measure of similarity between our original data vs batched dataset.
# 
# Briefly, SVCCA uses Singular Value Decomposition (SVD) to extract the components explaining 99% of the variation. This is done to remove potential dimensions described by noise. Next, SVCCA performs a Canonical Correlation Analysis (CCA) on the SVD matrices to identify maximum correlations of linear combinations of both input matrices. The algorithm will identify the canonical correlations of highest magnitude across and within algorithms of the same dimensionality.

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import os
import sys
import ast
import pandas as pd
import numpy as np
import random
import glob
import umap
import pickle
import warnings
warnings.filterwarnings(action='ignore')

sys.path.append("../")
from functions import cca_core

from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from numpy.random import seed
randomState = 123
seed(randomState)


# In[2]:


# Load config file
config_file = "config_exp_0.txt"

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


# Load data
base_dir = os.path.abspath(os.path.join(os.getcwd(),"../.."))

simulated_data_file = os.path.join(
    base_dir,
    "data",
    "simulated",
    analysis_name,
    "simulated_data.txt.xz")

batch_dir = os.path.join(
    base_dir,
    "data",
    "batch_simulated",
    analysis_name)

umap_model_file = umap_model_file = os.path.join(
    base_dir,
    "models",  
    NN_architecture,
    "umap_model.pkl")


# In[5]:


# Read in UMAP model
infile = open(umap_model_file, 'rb')
umap_model = pickle.load(infile)
infile.close()


# In[6]:


# Read in data
simulated_data = pd.read_table(
    simulated_data_file,
    header=0, 
    index_col=0,
    sep='\t')

simulated_data.head(10)


# ## Calculate Similarity using high dimensional (5K) batched data

# In[13]:


get_ipython().run_cell_magic('time', '', '# Calculate similarity using SVCCA\n\n# Store svcca scores\noutput_list = []\n\nfor i in num_batches:\n    print(\'Calculating SVCCA score for 1 batch vs {} batches..\'.format(i))\n    \n    # Get batch 1\n    batch_1_file = os.path.join(\n        batch_dir,\n        "Batch_1.txt.xz")\n\n    batch_1 = pd.read_table(\n        batch_1_file,\n        header=0,\n        index_col=0,\n        sep=\'\\t\')\n\n    # Use trained model to encode expression data into SAME latent space\n    original_data_df =  batch_1\n    \n    # All batches\n    batch_other_file = os.path.join(\n        batch_dir,\n        "Batch_"+str(i)+".txt.xz")\n\n    batch_other = pd.read_table(\n        batch_other_file,\n        header=0,\n        index_col=0,\n        sep=\'\\t\')\n    \n    print("Using batch {}".format(i))\n    \n    # Use trained model to encode expression data into SAME latent space\n    batch_data_df =  batch_other\n    \n    # Check shape: ensure that the number of samples is the same between the two datasets\n    if original_data_df.shape[0] != batch_data_df.shape[0]:\n        diff = original_data_df.shape[0] - batch_data_df.shape[0]\n        original_data_df = original_data_df.iloc[:-diff,:]\n    \n    # SVCCA\n    svcca_results = cca_core.get_cca_similarity(original_data_df.T,\n                                          batch_data_df.T,\n                                          verbose=False)\n    \n    output_list.append(np.mean(svcca_results["cca_coef1"]))\n\n# Convert output to pandas dataframe\nsvcca_raw_df = pd.DataFrame(output_list, columns=["svcca_mean_similarity"], index=num_batches)\nsvcca_raw_df')


# In[8]:


get_ipython().run_cell_magic('time', '', '# Permute simulated data\nshuffled_simulated_arr = []\nnum_samples = simulated_data.shape[0]\n\nfor i in range(num_samples):\n    row = list(simulated_data.values[i])\n    shuffled_simulated_row = random.sample(row, len(row))\n    shuffled_simulated_arr.append(shuffled_simulated_row)')


# In[9]:


get_ipython().run_cell_magic('time', '', '# SVCCA\nsvcca_results = cca_core.get_cca_similarity(simulated_data.T,\n                                      shuffled_simulated_data.T,\n                                      verbose=False)\n\nprint(np.mean(svcca_results["cca_coef1"]))')


# In[10]:


# Plot
svcca_raw_df.plot()


# ## Calculate Similarity using PCA projection of batched data

# In[ ]:


"""
output_list = []

for i in num_batches:
    print('Calculating SVCCA score for 1 batch vs {} batches..'.format(i))
    
    # Get batch 1
    batch_1_file = os.path.join(
        batch_dir,
        "Batch_1.txt")

    batch_1 = pd.read_table(
        batch_1_file,
        header=0,
        sep='\t',
        index_col=0)

    # PCA projection
    pca = PCA(n_components=num_PCs)

    # Use trained model to encode expression data into SAME latent space
    original_data_PCAencoded = pca.fit_transform(batch_1)


    original_data_PCAencoded_df = pd.DataFrame(original_data_PCAencoded,
                                         index=batch_1.index
                                         )
    
    # All batches
    batch_other_file = os.path.join(
        batch_dir,
        "Batch_"+str(i)+".txt")

    batch_other = pd.read_table(
        batch_other_file,
        header=0,
        sep='\t',
        index_col=0)
    
    print("Using batch {}".format(i))
    
    # Use trained model to encode expression data into SAME latent space
    batch_data_PCAencoded = pca.fit_transform(batch_other)
    
    
    batch_data_PCAencoded_df = pd.DataFrame(batch_data_PCAencoded,
                                         index=batch_other.index
                                         )
        
    # Check shape
    if original_data_PCAencoded_df.shape[0] != batch_data_PCAencoded_df.shape[0]:
        diff = original_data_PCAencoded_df.shape[0] - batch_data_PCAencoded_df.shape[0]
        original_data_PCAencoded_df = original_data_PCAencoded_df.iloc[:-diff,:]
    
    # SVCCA
    svcca_results = cca_core.get_cca_similarity(original_data_PCAencoded_df.T,
                                          batch_data_PCAencoded_df.T,
                                          verbose=False)
    
    output_list.append(np.mean(svcca_results["cca_coef1"]))

# Convert output to pandas dataframe
svcca_pca_df = pd.DataFrame(output_list, columns=["svcca_mean_similarity"], index=num_batches)
svcca_pca_df
"""


# In[ ]:


"""
# Plot
svcca_pca_df.plot()"""


# ## Manually compute similarity by applying CCA to PC batched data

# In[ ]:


"""
cca = CCA(n_components=1)

output_list = []

for i in num_batches:
    print('Calculating SVCCA score for 1 batch vs {} batches..'.format(i))
    
    # Get batch 1
    batch_1_file = os.path.join(
        batch_dir,
        "Batch_1.txt")

    batch_1 = pd.read_table(
        batch_1_file,
        header=0,
        sep='\t',
        index_col=0)

    # PCA projection
    pca = PCA(n_components=num_PCs)

    # Use trained model to encode expression data into SAME latent space
    original_data_PCAencoded = pca.fit_transform(batch_1)


    original_data_PCAencoded_df = pd.DataFrame(original_data_PCAencoded,
                                         index=batch_1.index
                                         )
    
    # All batches
    batch_other_file = os.path.join(
        batch_dir,
        "Batch_"+str(i)+".txt")

    batch_other = pd.read_table(
        batch_other_file,
        header=0,
        sep='\t',
        index_col=0)
    
    print("Using batch {}".format(i))
    
    # Use trained model to encode expression data into SAME latent space
    batch_data_PCAencoded = pca.fit_transform(batch_other)
    
    
    batch_data_PCAencoded_df = pd.DataFrame(batch_data_PCAencoded,
                                         index=batch_other.index
                                         )
        
    # Check shape
    if original_data_PCAencoded_df.shape[0] != batch_data_PCAencoded_df.shape[0]:
        diff = original_data_PCAencoded_df.shape[0] - batch_data_PCAencoded_df.shape[0]
        original_data_PCAencoded_df = original_data_PCAencoded_df.iloc[:-diff,:]
    
    # CCA
    U_c, V_c = cca.fit_transform(original_data_PCAencoded_df, batch_data_PCAencoded_df)
    result = np.corrcoef(U_c.T, V_c.T)[0,1]
    
    output_list.append(result)

# Convert output to pandas dataframe
pca_cca_df = pd.DataFrame(output_list, columns=["svcca_mean_similarity"], index=num_batches)
pca_cca_df
"""


# In[ ]:


"""
# Plot
pca_cca_df.plot()"""

