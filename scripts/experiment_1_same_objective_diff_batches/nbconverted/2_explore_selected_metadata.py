
# coding: utf-8

# # Explore selected biological signal
# 
# Ensure that the biological signal selected (i.e. PAO1 vs PA14, treatment vs no treatment) have a clear signal - clear separation in VAE latent space

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import os
import pandas as pd
import numpy as np
import random
import glob
from keras.models import model_from_json, load_model
import umap
import warnings
warnings.filterwarnings(action='once')

from ggplot import *

from numpy.random import seed
randomState = 123
seed(randomState)


# In[2]:


# Parameters
NN_architecture = 'NN_2500_20'
metadata_field = 'strain'


# In[3]:


# Load arguments
base_dir = os.path.abspath(os.path.join(os.getcwd(),"../.."))

mapping_file = os.path.join(
    base_dir,
    "data",
    "metadata",
    "mapping_{}.txt".format(metadata_field))

normalized_data_file = os.path.join(
    base_dir,
    "data",
    "input",
    "train_set_normalized.pcl")

encoded_data_file = glob.glob(os.path.join(
    base_dir,
    "data",
    "encoded",
    NN_architecture,
    "*encoded.txt"))[0]


# In[4]:


# Read data
normalized_data = pd.read_table(
    normalized_data_file,
    header=0,
    sep='\t',
    index_col=0).T

print(normalized_data.shape)
normalized_data.head(10)


# In[5]:


# Read encoded data
encoded_data = pd.read_table(
    encoded_data_file,
    header=0,
    sep='\t',
    index_col=0)

print(encoded_data.shape)
encoded_data.head(10)


# In[6]:


# Read in metadata
metadata = pd.read_table(
    mapping_file, 
    header=0, 
    sep='\t', 
    index_col=0)

metadata_field = metadata.columns[0]

metadata.head(10)


# ## Plot input data using UAMP

# In[7]:


# Merge gene expression data and metadata
data_labeled = normalized_data.merge(
    metadata,
    left_index=True, 
    right_index=True, 
    how='inner')

print(data_labeled.shape)
data_labeled.head(5)


# In[8]:


# UMAP embedding of raw gene space data
embedding = umap.UMAP().fit_transform(data_labeled.iloc[:,1:-1])
embedding_df = pd.DataFrame(data=embedding, columns=['1','2'])
embedding_df['metadata'] = list(data_labeled[metadata_field])
print(embedding_df.shape)
embedding_df.head(5)


# In[9]:


# Replace NaN with string "NA"
embedding_df['metadata'] = embedding_df.metadata.fillna('NA')


# In[10]:


# Plot
ggplot(aes(x='1',y='2', color='metadata'), data=embedding_df) +         geom_point(alpha=0.5) +         scale_color_brewer(type='qual', palette='Set1') +         ggtitle("Input data")


# ## Plot encoded data using UMAP

# In[11]:


# Merge gene expression data and metadata
data_encoded_labeled = encoded_data.merge(
    metadata,
    left_index=True, 
    right_index=True, 
    how='inner')

print(data_encoded_labeled.shape)
data_encoded_labeled.head(5)


# In[12]:


# UMAP embedding of raw gene space data
embedding_encoded = umap.UMAP().fit_transform(data_encoded_labeled.iloc[:,1:-1])
embedding_encoded_df = pd.DataFrame(data=embedding_encoded, columns=['1','2'])
embedding_encoded_df['metadata'] = list(data_encoded_labeled[metadata_field])
print(embedding_encoded_df.shape)
embedding_encoded_df.head(5)


# In[13]:


# Replace NaN with string "NA"
embedding_encoded_df['metadata'] = embedding_encoded_df.metadata.fillna('NA')


# In[14]:


# Plot
ggplot(aes(x='1',y='2', color='metadata'), data=embedding_encoded_df) +         geom_point(alpha=0.5) +         scale_color_brewer(type='qual', palette='Set1') +         ggtitle("Encoded data")

