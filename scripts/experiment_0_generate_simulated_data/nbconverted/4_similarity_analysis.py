
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
from sklearn.cross_decomposition import CCA
from numpy.random import seed
randomState = 123
seed(randomState)


# In[2]:


# Parameters
analysis_name = 'experiment_0'
NN_architecture = 'NN_2500_30'
num_PCs = 100
num_batches = [1,2,5,10,20,50,100,500,1000,2000,3000,6000]
#num_batches = [1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50]
#num_batches = [1,2,3,4,5,6,7,8,9,10,15,20,50,100,500,800,1000]


# In[3]:


# Load data
base_dir = os.path.abspath(os.path.join(os.getcwd(),"../.."))

simulated_data_file = os.path.join(
    base_dir,
    "data",
    "simulated",
    analysis_name,
    "simulated_data.txt")

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


# In[4]:


# Read in UMAP model
infile = open(umap_model_file, 'rb')
umap_model = pickle.load(infile)
infile.close()


# In[5]:


# Read in data
simulated_data = pd.read_table(
    simulated_data_file,
    header=0, 
    index_col=0,
    sep='\t')

simulated_data.head(10)


# ## Calculate Similarity using high dimensional (5K) batched data

# In[6]:


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
    original_data_df =  batch_1
    
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
    batch_data_df =  batch_other
    
    # Check shape
    if original_data_df.shape[0] != batch_data_df.shape[0]:
        diff = original_data_df.shape[0] - batch_data_df.shape[0]
        original_data_df = original_data_df.iloc[:-diff,:]
    
    # SVCCA
    svcca_results = cca_core.get_cca_similarity(original_data_df.T,
                                          batch_data_df.T,
                                          verbose=False)
    
    output_list.append(np.mean(svcca_results["cca_coef1"]))

# Convert output to pandas dataframe
svcca_raw_df = pd.DataFrame(output_list, columns=["svcca_mean_similarity"], index=num_batches)
svcca_raw_df


# In[7]:


# Plot
svcca_raw_df.plot()


# ## Check positive control
# We want to verify that SVCCA(input, input) = 1.  This does not seem to be the case, but I'm not sure why

# In[8]:


# Check datasets batch 1 and original are the same
batch_1_file = os.path.join(
    batch_dir,
    "Batch_1.txt")
print(batch_1_file)

batch_1 = pd.read_table(
    batch_1_file,
    header=0,
    sep='\t',
    index_col=0)
    
batch_1.head(10)


# In[9]:


i = 1
batch_other_file = os.path.join(
    batch_dir,
    "Batch_"+str(i)+".txt")
print(batch_other_file)

batch_other = pd.read_table(
    batch_other_file,
    header=0,
    sep='\t',
    index_col=0)

batch_other.head(10)


# In[10]:


# Calculate SVCCA(batch_1, batch_1)
svcca_results_batch1_itself = cca_core.get_cca_similarity(batch_1.T,
                                          batch_1.T,
                                          verbose=False)
np.mean(svcca_results_batch1_itself["cca_coef1"])


# In[11]:


# Caluclate SVCCA(batch_1, batch_1 variable)
svcca_results_batch1_other = cca_core.get_cca_similarity(batch_1.T,
                                          batch_other.T,
                                          verbose=False)
np.mean(svcca_results_batch1_other["cca_coef1"])


# ## Calculate Similarity using PCA projection of batched data

# In[24]:


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


# In[25]:


# Plot
svcca_pca_df.plot()


# ## Manually compute similarity by applying CCA to PC batched data

# In[18]:


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


# In[19]:


# Plot
pca_cca_df.plot()

