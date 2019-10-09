
# coding: utf-8

# # Visualize data
# 
# In order to start making interpretations we will generate two visualizations of our data
# 
# 1. We will verify that the simulated dataset is a good representation of our original input dataset by visually comparing the structures in the two datasets projected onto UMAP space.
# 
# 2. We will plot the PCA projected data after adding experiments to examine how the technical variation shifted the data.
# 
# 3. We plot the PCA projected data after correcting for the technical variation introduced by the experiments and examine the effectiveness of the correction method by comparing the data before and after the correction.

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import os
import ast
import pandas as pd
import numpy as np
import random
import glob
from plotnine import ggplot, ggtitle, xlab, ylab, geom_point, aes, facet_wrap, scale_color_manual, xlim, ylim, scale_color_brewer 
from sklearn.decomposition import PCA
from keras.models import load_model
import umap

import warnings
warnings.filterwarnings(action='ignore')

from numpy.random import seed
randomState = 123
seed(randomState)


# In[2]:


# User parameters
NN_architecture = 'NN_2500_30'
analysis_name = 'analysis_1'


# In[3]:


# Load data

# base dir on repo
base_dir = os.path.abspath(os.path.join(os.getcwd(),"../.."))          

# base dir on local machine for data storage
# os.makedirs doesn't recognize `~`
local_dir = os.path.abspath(os.path.join(os.getcwd(), "../../../.."))  
    
latent_dim = NN_architecture.split('_')[-1]

NN_dir = base_dir + "/models/" + NN_architecture

normalized_data_file = os.path.join(
    base_dir,
    "data",
    "input",
    "train_set_normalized.pcl")

model_encoder_file = glob.glob(os.path.join( ## Make more explicit name here
    NN_dir,
    "tybalt_2layer_{}latent_encoder_model.h5".format(latent_dim)))[0]

weights_encoder_file = glob.glob(os.path.join(
    NN_dir,
    "tybalt_2layer_{}latent_encoder_weights.h5".format(latent_dim)))[0]

model_decoder_file = glob.glob(os.path.join(
    NN_dir,
    "tybalt_2layer_{}latent_decoder_model.h5".format(latent_dim)))[0]

weights_decoder_file = glob.glob(os.path.join(
    NN_dir,
    "tybalt_2layer_{}latent_decoder_weights.h5".format(latent_dim)))[0]

experiment_dir = os.path.join(
    local_dir,
    "Data",
    "Batch_effects",
    "experiment_simulated",
    analysis_name)

simulated_data_file = os.path.join(
    local_dir,
    "Data",
    "Batch_effects",
    "simulated",
    analysis_name,
    "simulated_data.txt.xz")

permuted_simulated_data_file = os.path.join(
    local_dir,
    "Data",
    "Batch_effects",
    "simulated",
    analysis_name,
    "permuted_simulated_data.txt.xz")


# ## 1. Visualize simulated data in UMAP space

# In[4]:


# Load saved models
loaded_model = load_model(model_encoder_file)
loaded_decode_model = load_model(model_decoder_file)

loaded_model.load_weights(weights_encoder_file)
loaded_decode_model.load_weights(weights_decoder_file)


# In[5]:


# Read data
normalized_data = pd.read_table(
    normalized_data_file,
    header=0,
    sep='\t',
    index_col=0).T

simulated_data = pd.read_table(
    simulated_data_file,
    header=0,
    sep='\t',
    index_col=0)

print(normalized_data.shape)
print(simulated_data.shape)


# In[6]:


normalized_data.head(10)


# In[7]:


simulated_data.head(10)


# In[8]:


# Add labels to original normalized data
sample_ids = list(simulated_data.index)


normalized_data_label = normalized_data.copy()

normalized_data_label['color_by'] = 'Not in experiment'
normalized_data_label.loc[sample_ids, 'color_by'] = simulated_data['color_by']
normalized_data_label.loc[sample_ids].head(10)


# In[22]:


# UMAP embedding of original input data

# Get and save model
model = umap.UMAP(random_state=randomState).fit(normalized_data)

input_data_UMAPencoded = model.transform(normalized_data)
input_data_UMAPencoded_df = pd.DataFrame(data=input_data_UMAPencoded,
                                         index=normalized_data.index,
                                         columns=['1','2'])
# Add label
input_data_UMAPencoded_df['color_by'] = normalized_data_label['color_by']

ggplot(input_data_UMAPencoded_df, aes(x='1',y='2'))     + geom_point(aes(color='color_by'), alpha=1)     + xlim(7, 11)     + ylim(-3, -2.5)     + ggtitle('Input data')


# In[28]:


label_ls = normalized_data_label['color_by'].unique()


# In[29]:


# UMAP embedding of simulated data

# Drop label column
simulated_data_numeric = simulated_data.drop(['color_by'], axis=1)

simulated_data_UMAPencoded = model.transform(simulated_data_numeric)
simulated_data_UMAPencoded_df = pd.DataFrame(data=simulated_data_UMAPencoded,
                                         index=simulated_data.index,
                                         columns=['1','2'])

# Add back label column
simulated_data_UMAPencoded_df['color_by'] = simulated_data['color_by']


ggplot(simulated_data_UMAPencoded_df, aes(x='1',y='2'))     + geom_point(aes(color='color_by'), alpha=1)     #+ scale_color_brewer(type='qual', palette='Set1', labels=label_ls) \
    + ggtitle("Simulated data")


# ### Side by side view

# In[ ]:


# Add label for input or simulated dataset
input_data_UMAPencoded_df['dataset'] = 'original'
simulated_data_UMAPencoded_df['dataset'] = 'simulated'

# Concatenate input and simulated dataframes together
combined_data_df = pd.concat([input_data_UMAPencoded_df, simulated_data_UMAPencoded_df])

# Plot
ggplot(combined_data_df, aes(x='1', y='2')) + geom_point(aes(color='color_by'), alpha=1) + facet_wrap('~dataset') + xlab('UMAP 1') + ylab('UMAP 2') + ggtitle('UMAP of original and simulated data')


# In[ ]:


# Zoomed in view

# Add label for input or simulated dataset
input_data_UMAPencoded_df['dataset'] = 'original'
simulated_data_UMAPencoded_df['dataset'] = 'simulated'

# Concatenate input and simulated dataframes together
combined_data_df = pd.concat([input_data_UMAPencoded_df, simulated_data_UMAPencoded_df])

# Plot
ggplot(combined_data_df, aes(x='1', y='2')) + geom_point(aes(color='color_by'), alpha=1) + facet_wrap('~dataset') + xlab('UMAP 1') + ylab('UMAP 2') + xlim(7, 11) + ylim(-4, 0) + ggtitle('UMAP of original and simulated data')

