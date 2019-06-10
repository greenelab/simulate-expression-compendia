
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
import pandas as pd
import numpy as np
import random
import glob
import umap
import pickle
import warnings
warnings.filterwarnings(action='once')

from ggplot import *
from functions import cca_core
from sklearn.decomposition import PCA
from numpy.random import seed
randomState = 123
seed(randomState)


# In[2]:


# Parameters
analysis_name = 'full_dataset'
NN_architecture = 'NN_2500_300'
num_batches = [1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50]


# In[3]:


# Load data
batch_dir = os.path.join(
    os.path.dirname(os.getcwd()),
    "data",
    "batch_simulated",
    analysis_name)

umap_model_file = umap_model_file = os.path.join(
    os.path.dirname(os.getcwd()),
    "models",  
    NN_architecture,
    "umap_model.pkl")


# In[4]:


# Read in UMAP model
infile = open(umap_model_file, 'rb')
umap_model = pickle.load(infile)
infile.close()


# In[5]:


# Calculate Similarity using UMAP representation of batched data

output_list = []

for i in num_batches:
    print('Calculating SVCCA score for 1 batch vs {} batches..'.format(i))
    if i ==1:
        batch_data_file = os.path.join(
            batch_dir,
            "Batch_"+str(i)+".txt")

        batch_data = pd.read_table(
            batch_data_file,
            header=0,
            sep='\t',
            index_col=0)

        # UMAP embedding of decoded batch data
        original_data_UMAPencoded = umap_model.transform(batch_data)
        original_data_UMAPencoded_df = pd.DataFrame(data=original_data_UMAPencoded,
                                                 index=batch_data.index,
                                                 columns=['1','2'])
    batch_file = os.path.join(
        batch_dir,
        "Batch_"+str(i)+".txt")

    batch_data = pd.read_table(
        batch_data_file,
        header=0,
        sep='\t',
        index_col=0)

    # UMAP embedding of decoded batch data
    batch_data_UMAPencoded = umap.UMAP().fit_transform(batch_data)
    batch_data_UMAPencoded_df = pd.DataFrame(data=batch_data_UMAPencoded,
                                             index=batch_data.index,
                                             columns=['1','2'])

    # SVCCA
    svcca_results = cca_core.get_cca_similarity(original_data_UMAPencoded_df.T,
                                          batch_data_UMAPencoded_df.T,
                                          verbose=False)
    
    output_list.append(np.mean(svcca_results["cca_coef1"]))

# Convert output to pandas dataframe
svcca_umap_df = pd.DataFrame(output_list, columns=["svcca_mean_similarity"], index=num_batches)
svcca_umap_df


# In[6]:


# Calculate Similarity using PCA projection of batched data

output_list = []

for i in num_batches:
    print('Calculating SVCCA score for 1 batch vs {} batches..'.format(i))
    if i ==1:
        batch_data_file = os.path.join(
            batch_dir,
            "Batch_"+str(i)+".txt")

        batch_data = pd.read_table(
            batch_data_file,
            header=0,
            sep='\t',
            index_col=0)

        # PCA projection
        num_PCs = 2
        pca = PCA(n_components=num_PCs)

        # Use trained model to encode expression data into SAME latent space
        original_data_PCAencoded = pca.fit_transform(batch_data)


        original_data_PCAencoded_df = pd.DataFrame(original_data_PCAencoded,
                                             index=batch_data.index,
                                             columns=['1', '2'])
    
    # All batches
    batch_file = os.path.join(
        batch_dir,
        "Batch_"+str(i)+".txt")

    batch_data = pd.read_table(
        batch_data_file,
        header=0,
        sep='\t',
        index_col=0)

    # PCA projection
    num_PCs = 2
    pca = PCA(n_components=num_PCs)

    # Use trained model to encode expression data into SAME latent space
    batch_data_PCAencoded = pca.fit_transform(batch_data)
    
    
    batch_data_PCAencoded_df = pd.DataFrame(batch_data_PCAencoded,
                                         index=batch_data.index,
                                         columns=['1', '2'])

    # SVCCA
    svcca_results = cca_core.get_cca_similarity(original_data_PCAencoded_df.T,
                                          batch_data_PCAencoded_df.T,
                                          verbose=False)
    
    output_list.append(np.mean(svcca_results["cca_coef1"]))

# Convert output to pandas dataframe
svcca_pca_df = pd.DataFrame(output_list, columns=["svcca_mean_similarity"], index=num_batches)
svcca_pca_df

