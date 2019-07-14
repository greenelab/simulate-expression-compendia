
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
import pickle
import warnings
warnings.filterwarnings(action='once')

from ggplot import *

from numpy.random import seed
randomState = 123
seed(randomState)


# In[2]:


# Parameters
NN_architecture = 'NN_300_10'
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

# Saved models
loaded_model = load_model(model_encoder_file)
loaded_decode_model = load_model(model_decoder_file)

loaded_model.load_weights(weights_encoder_file)
loaded_decode_model.load_weights(weights_decoder_file)

# Output
umap_model_file = os.path.join(
    base_dir,
    "models",  
    NN_architecture,
    "umap_model.pkl")


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


# ## Plot input data using UMAP

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


# Get and save model
model = umap.UMAP(random_state=randomState).fit(normalized_data)
pickle.dump(model, open(umap_model_file, 'wb'))


# In[9]:


# UMAP embedding of raw gene space data
embedding = model.transform(data_labeled.iloc[:,:-1])
embedding_df = pd.DataFrame(data=embedding, columns=['1','2'])
embedding_df['metadata'] = list(data_labeled[metadata_field])
print(embedding_df.shape)
embedding_df.head(5)


# In[10]:


# Replace NaN with string "NA"
embedding_df['metadata'] = embedding_df.metadata.fillna('NA')


# In[11]:


# Plot
if metadata_field == 'treatment':
    color_theme = 'Set3'
else:
    color_theme = 'Set1'
    
ggplot(aes(x='1',y='2', color='metadata'), data=embedding_df) +         geom_point(alpha=0.5) +         scale_color_brewer(type='qual', palette=color_theme) +         ggtitle("Input data")


# ## Plot encoded data using UMAP

# In[12]:


# Merge gene expression data and metadata
data_encoded_labeled = encoded_data.merge(
    metadata,
    left_index=True, 
    right_index=True, 
    how='inner')

print(data_encoded_labeled.shape)
data_encoded_labeled.head(5)


# In[13]:


# UMAP embedding of encoded data
embedding_encoded = umap.UMAP(random_state=randomState).fit_transform(data_encoded_labeled.iloc[:,:-1])
embedding_encoded_df = pd.DataFrame(data=embedding_encoded, columns=['1','2'])
embedding_encoded_df['metadata'] = list(data_encoded_labeled[metadata_field])
print(embedding_encoded_df.shape)
embedding_encoded_df.head(5)


# In[14]:


# Replace NaN with string "NA"
embedding_encoded_df['metadata'] = embedding_encoded_df.metadata.fillna('NA')


# In[15]:


# Plot
ggplot(aes(x='1',y='2', color='metadata'), data=embedding_encoded_df) +         geom_point(alpha=0.5) +         scale_color_brewer(type='qual', palette=color_theme) +         ggtitle("Encoded data")


# ## Plot decoded data using UMAP

# In[16]:


# Decode data back into gene space
data_decoded = loaded_decode_model.predict_on_batch(encoded_data)
data_decoded_df = pd.DataFrame(data_decoded, index=encoded_data.index)


# In[17]:


# Merge gene expression data and metadata
data_decoded_labeled = data_decoded_df.merge(
    metadata,
    left_index=True, 
    right_index=True, 
    how='inner')

print(data_decoded_labeled.shape)
data_decoded_labeled.head(5)


# In[18]:


# UMAP embedding of decoded data
embedding_decoded = model.transform(data_decoded_labeled.iloc[:,:-1])
embedding_decoded_df = pd.DataFrame(data=embedding_decoded, columns=['1','2'])
embedding_decoded_df['metadata'] = list(data_decoded_labeled[metadata_field])
print(embedding_decoded_df.shape)
embedding_decoded_df.head(5)


# In[19]:


# Replace NaN with string "NA"
embedding_decoded_df['metadata'] = embedding_decoded_df.metadata.fillna('NA')


# In[20]:


# Plot
ggplot(aes(x='1',y='2', color='metadata'), data=embedding_decoded_df) +         geom_point(alpha=0.5) +         scale_color_brewer(type='qual', palette=color_theme) +         ggtitle("Decoded data")

