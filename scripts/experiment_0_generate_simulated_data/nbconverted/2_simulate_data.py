
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
import pickle
from keras.models import model_from_json, load_model
from ggplot import *
import umap
import warnings
warnings.filterwarnings(action='once')

from numpy.random import seed
randomState = 123
seed(randomState)


# In[2]:


# Parameters
analysis_name = 'experiment_0'
NN_architecture = 'NN_2500_30'
num_simulated_samples = 1000


# In[3]:


# Create directories
base_dir = os.path.abspath(os.path.join(os.getcwd(),"../.."))

new_dir = os.path.join(base_dir, "data", "simulated")

analysis_dir = os.path.join(new_dir, analysis_name)

if os.path.exists(analysis_dir):
    print('directory already exists: {}'.format(analysis_dir))
else:
    print('creating new directory: {}'.format(analysis_dir))
os.makedirs(analysis_dir, exist_ok=True)


# In[4]:


# Load arguments
normalized_data_file = os.path.join(
    base_dir,
    "data",
    "input",
    "train_set_normalized.pcl")

metadata_file = os.path.join(
    base_dir,
    "data",
    "metadata",
    "sample_annotations.tsv")

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
simulated_data_file = os.path.join(
    base_dir,
    "data",
    "simulated",
    analysis_name,
    "simulated_data.txt")

umap_model_file = os.path.join(
    base_dir,
    "models",  
    NN_architecture,
    "umap_model.pkl")


# In[5]:


# Read data
normalized_data = pd.read_table(
    normalized_data_file,
    header=0,
    sep='\t',
    index_col=0).T

print(normalized_data.shape)
normalized_data.head(10)


# ## Plot input data using UMAP

# In[6]:


# UMAP embedding

# Get and save model
model = umap.UMAP(random_state=randomState).fit(normalized_data)
pickle.dump(model, open(umap_model_file, 'wb'))

input_data_UMAPencoded = model.transform(normalized_data)
input_data_UMAPencoded_df = pd.DataFrame(data=input_data_UMAPencoded,
                                         index=normalized_data.index,
                                         columns=['1','2'])


g = ggplot(aes(x='1',y='2'), data=input_data_UMAPencoded_df) +             geom_point(alpha=0.5) +             scale_color_brewer(type='qual', palette='Set2') +             scale_x_continuous(limits=(-15,20)) +            scale_y_continuous(limits=(-15,15)) +             ggtitle("Input data")

print(g)


# ## Plot encoded input data using UMAP

# In[7]:


# Encode data into latent space
data_encoded = loaded_model.predict_on_batch(normalized_data)
data_encoded_df = pd.DataFrame(data_encoded, index=normalized_data.index)

# Plot
latent_data_UMAPencoded = umap.UMAP(random_state=randomState).fit_transform(data_encoded_df)
latent_data_UMAPencoded_df = pd.DataFrame(data=latent_data_UMAPencoded,
                                         index=data_encoded_df.index,
                                         columns=['1','2'])


g = ggplot(aes(x='1',y='2'), data=latent_data_UMAPencoded_df) +             geom_point(alpha=0.5) +             scale_color_brewer(type='qual', palette='Set2') +             ggtitle("Encoded input data")

print(g)


# ## Plot decoded input data using UMAP

# In[8]:


# Decode data back into gene space
data_decoded = loaded_decode_model.predict_on_batch(data_encoded_df)
data_decoded_df = pd.DataFrame(data_decoded, index=data_encoded_df)

# Plot
data_decoded_UMAPencoded = model.transform(data_decoded_df)
data_decoded_UMAPencoded_df = pd.DataFrame(data=data_decoded_UMAPencoded,
                                         index=data_decoded_df.index,
                                         columns=['1','2'])


g = ggplot(aes(x='1',y='2'), data=data_decoded_UMAPencoded_df) +             geom_point(alpha=0.5) +             scale_color_brewer(type='qual', palette='Set2') +             ggtitle("Decoded input data")

print(g)


# ## Simulate data
# 
# Generate new simulated data by sampling from the distribution of latent space features.  In other words, for each latent space feature get the mean and standard deviation.  Then we can generate a new sample by sampling from a distribution with this mean and standard deviation.

# In[9]:


# Simulate data

# Encode into latent space
data_encoded = loaded_model.predict_on_batch(normalized_data)
data_encoded_df = pd.DataFrame(data_encoded, index=normalized_data.index)

latent_dim = data_encoded_df.shape[1]

# Get mean and standard deviation per encoded feature
encoded_means = data_encoded_df.mean(axis=0)

encoded_stds = data_encoded_df.std(axis=0)

# Generate samples 
new_data = np.zeros([num_simulated_samples,latent_dim])
for j in range(latent_dim):
    # Use mean and std for feature
    new_data[:,j] = np.random.normal(encoded_means[j], encoded_stds[j], num_simulated_samples) 
    
    # Use standard normal
    #new_data[:,j] = np.random.normal(0, 1, num_simulated_samples)
    
new_data_df = pd.DataFrame(data=new_data)

# Decode N samples
new_data_decoded = loaded_decode_model.predict_on_batch(new_data_df)
new_data_decoded_df = pd.DataFrame(data=new_data_decoded)

new_data_decoded_df.head(10)


# ## Plot simulated data using UMAP
# 
# Note: we will use the same UMAP mapping for the input and simulated data to ensure they are plotted on the same space.

# In[10]:


# UMAP embedding
simulated_data_UMAPencoded = model.transform(new_data_decoded_df)
simulated_data_UMAPencoded_df = pd.DataFrame(data=simulated_data_UMAPencoded,
                                         index=new_data_decoded_df.index,
                                         columns=['1','2'])


g = ggplot(aes(x='1',y='2'), data=simulated_data_UMAPencoded_df) +             geom_point(alpha=0.5) +             scale_color_brewer(type='qual', palette='Set2') +             ggtitle("Simulated data")

print(g)


# In[11]:


# Output
new_data_decoded_df.to_csv(simulated_data_file, sep='\t')

