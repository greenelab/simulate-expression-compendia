
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
from keras.models import model_from_json, load_model
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
analysis_name = 'experiment_1'
NN_architecture = 'NN_2500_10'
#num_batches = [1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50]
num_batches = [1,2,3,4,5,6,7,8,9,10,15,20,50,100,500,800]
num_PCs = 5


# In[3]:


# Load data
base_dir = os.path.abspath(os.path.join(os.getcwd(),"../.."))

batch_dir = os.path.join(
    base_dir,
    "data",
    "batch_simulated",
    analysis_name)

umap_model_file = os.path.join(
    base_dir,
    "models",  
    NN_architecture,
    "umap_model.pkl")

encoded_data_file = glob.glob(os.path.join(
    base_dir,
    "data",
    "encoded",
    NN_architecture,
    "*encoded.txt"))[0]

model_encoder_file = glob.glob(os.path.join(
    base_dir,
    "models",
    NN_architecture,
    "*_encoder_model.h5"))[0]

weights_encoder_file = glob.glob(os.path.join(
    base_dir,
    "models",
    NN_architecture,
    "*_encoder_weights.h5"))[0]

model_decoder_file = glob.glob(os.path.join(
    base_dir,
    "models", 
    NN_architecture,
    "*_decoder_model.h5"))[0]


weights_decoder_file = glob.glob(os.path.join(
    base_dir,
    "models",  
    NN_architecture,
    "*_decoder_weights.h5"))[0]


# In[4]:


# Read in VAE models
loaded_model = load_model(model_encoder_file)
loaded_decode_model = load_model(model_decoder_file)

loaded_model.load_weights(weights_encoder_file)
loaded_decode_model.load_weights(weights_decoder_file)


# In[5]:


# Read in UMAP model
infile = open(umap_model_file, 'rb')
umap_model = pickle.load(infile)
infile.close()


# In[6]:


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

        # Encode data using VAE model
        data_encoded_batch1 = loaded_model.predict_on_batch(batch_data)
        data_encoded_batch1_df = pd.DataFrame(data_encoded_batch1, index=batch_data.index)
        
    batch_file = os.path.join(
        batch_dir,
        "Batch_"+str(i)+".txt")

    batch_data = pd.read_table(
        batch_data_file,
        header=0,
        sep='\t',
        index_col=0)

    # Encode data using VAE model
    data_encoded = loaded_model.predict_on_batch(batch_data)
    data_encoded_df = pd.DataFrame(data_encoded, index=batch_data.index)

    # SVCCA
    svcca_results = cca_core.get_cca_similarity(data_encoded_batch1_df.T,
                                          data_encoded_df.T,
                                          verbose=False)
    
    output_list.append(np.mean(svcca_results["cca_coef1"]))

# Convert output to pandas dataframe
svcca_umap_df = pd.DataFrame(output_list, columns=["svcca_mean_similarity"], index=num_batches)
svcca_umap_df


# In[7]:


# Calculate Similarity using PCA projection of batched data
# FIX COMPARISON

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
        pca = PCA(n_components=num_PCs)

        # Use trained model to encode expression data into SAME latent space
        original_data_PCAencoded = pca.fit_transform(batch_data)


        original_data_PCAencoded_df = pd.DataFrame(original_data_PCAencoded,
                                             index=batch_data.index)
    
    # All batches
    batch_file = os.path.join(
        batch_dir,
        "Batch_"+str(i)+".txt")

    batch_data = pd.read_table(
        batch_file,
        header=0,
        sep='\t',
        index_col=0)

    # PCA projection
    pca = PCA(n_components=num_PCs)

    # Use trained model to encode expression data into SAME latent space
    batch_data_PCAencoded = pca.fit_transform(batch_data)
    
    
    batch_data_PCAencoded_df = pd.DataFrame(batch_data_PCAencoded,
                                         index=batch_data.index)

    # SVCCA
    svcca_results = cca_core.get_cca_similarity(original_data_PCAencoded_df.T,
                                          batch_data_PCAencoded_df.T,
                                          verbose=False)
    
    output_list.append(np.mean(svcca_results["cca_coef1"]))

# Convert output to pandas dataframe
svcca_pca_df = pd.DataFrame(output_list, columns=["svcca_mean_similarity"], index=num_batches)
svcca_pca_df

