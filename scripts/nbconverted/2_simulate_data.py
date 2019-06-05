
# coding: utf-8

# # Generate simulated data
# 
# Generate simulated data by sampling from VAE latent sapce
# 
# Workflow:
# 1. Input gene expression data from 1 experiment (here we are assuming that there is only biological variation within this experiment)
# 2. Encode this input into a latent space using the trained VAE model
# 3. For each encoded feature, sample from a distribution using the the mean and standard deviation for that feature
# 4. Decode the samples

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import os
import pandas as pd
import numpy as np
import random
import glob
from keras.models import model_from_json, load_model
from ggplot import *
import umap

from numpy.random import seed
randomState = 123
seed(randomState)


# In[2]:


# Parameters
analysis_name = 'treatment'
experiment_id = 'E-GEOD-24036'
num_simulated_samples = 100


# In[3]:


# Create directories
new_dir = os.path.join(os.path.dirname(os.getcwd()), "data", "simulated")

analysis_dir = os.path.join(new_dir, analysis_name)

if os.path.exists(analysis_dir):
    print('directory already exists: {}'.format(analysis_dir))
else:
    print('creating new directory: {}'.format(analysis_dir))
os.makedirs(analysis_dir, exist_ok=True)


# In[4]:


# Load arguments
normalized_data_file = os.path.join(
    os.path.dirname(os.getcwd()),
    "data",
    "input",
    "train_set_normalized.pcl")

metadata_file = os.path.join(
    os.path.dirname(os.getcwd()),
    "data",
    "metadata",
    "sample_annotations.tsv")

model_encoder_file = glob.glob(os.path.join(
        os.path.dirname(os.getcwd()),
        "models",
        "*_encoder_model.h5"))[0]

weights_encoder_file = glob.glob(os.path.join(
    os.path.dirname(os.getcwd()),
    "models",
    "*_encoder_weights.h5"))[0]

model_decoder_file = glob.glob(os.path.join(
    os.path.dirname(os.getcwd()),
    "models", 
    "*_decoder_model.h5"))[0]


weights_decoder_file = glob.glob(os.path.join(
    os.path.dirname(os.getcwd()),
    "models",  
    "*_decoder_weights.h5"))[0]

# Output
simulated_data_file = os.path.join(
    os.path.dirname(os.getcwd()),
    "data",
    "simulated",
    analysis_name,
    "simulated_data.txt")


# In[5]:


# Read data
normalized_data = pd.read_table(normalized_data_file, header=0, sep='\t', index_col=0).T
normalized_data.shape


# In[6]:


# Read in metadata
metadata = pd.read_table(metadata_file, header=0, sep='\t', index_col='ml_data_source')
metadata


# In[7]:


# read in saved models
loaded_model = load_model(model_encoder_file)
loaded_decode_model = load_model(model_decoder_file)

# load weights into new model
loaded_model.load_weights(weights_encoder_file)
loaded_decode_model.load_weights(weights_decoder_file)


# In[8]:


# Metadata mapping
metadata_field = 'experiment'
metadata_map = metadata[metadata_field].to_frame()

metadata_map.head(5)


# ## Select experiment(s)

# In[9]:


# Select input experiment
selected_samples = list(metadata_map[metadata_map['experiment'] == experiment_id].index)

print(len(selected_samples))

data_selected = normalized_data.loc[selected_samples]
data_selected.head(5)


# ## Plot input data using UMAP

# In[10]:


# UMAP embedding of selected data
input_data_UMAPencoded = umap.UMAP().fit_transform(data_selected)
input_data_UMAPencoded_df = pd.DataFrame(data=input_data_UMAPencoded,
                                         index=data_selected.index,
                                         columns=['1','2'])


g = ggplot(aes(x='1',y='2'), data=input_data_UMAPencoded_df) +             geom_point(alpha=0.5) +             scale_color_brewer(type='qual', palette='Set2') +             ggtitle("Input data")

print(g)


# ## Simulate data
# 
# Generate new simulated data by sampling from the distribution of latent space features.  In other words, for each latent space feature get the mean and standard deviation.  Then we can generate a new sample by sampling from a distribution with this mean and standard deviation.

# In[11]:


# Simulate data

# Encode into latent space
data_selected_encoded = loaded_model.predict_on_batch(data_selected)
data_selected_encoded_df = pd.DataFrame(data_selected_encoded, index=data_selected.index)

latent_dim = data_selected_encoded_df.shape[1]

# Get mean and standard deviation per encoded feature
encoded_means = data_selected_encoded_df.mean(axis=0)

encoded_stds = data_selected_encoded_df.std(axis=0)

# Generate samples
new_data = np.zeros([num_simulated_samples,latent_dim])
for j in range(latent_dim):
    new_data[:,j] = np.random.normal(encoded_means[j], encoded_stds[j], num_simulated_samples) 

new_data_df = pd.DataFrame(data=new_data)

# Decode N samples
new_data_decoded = loaded_decode_model.predict_on_batch(new_data_df)
new_data_decoded_df = pd.DataFrame(data=new_data_decoded)

new_data_decoded_df.head(10)


# In[12]:


# Output
new_data_decoded_df.to_csv(simulated_data_file, sep='\t')

