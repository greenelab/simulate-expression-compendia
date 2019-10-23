
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
from plotnine import ggplot, ggtitle, xlab, ylab, geom_point, aes, facet_wrap, scale_color_manual, xlim, ylim, scale_colour_manual 
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


# ## Load data

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


# Remove corresponding rows from simulated_data df
sample_ids = list(simulated_data.index)

sample_ids_toremove = []
for sample_id in sample_ids:
    if sample_id not in list(normalized_data.index):
        sample_ids_toremove.append(sample_id)
       
simulated_data.drop(sample_ids_toremove, inplace=True)


# In[9]:


# Remove any sample_ids that are not found in gene expression data
sample_ids = [sample for sample in sample_ids if sample in normalized_data.index]


# In[10]:


# Add labels to original normalized data
normalized_data_label = normalized_data.copy()

normalized_data_label['experiment_id'] = 'Not selected'
normalized_data_label.loc[sample_ids, 'experiment_id'] = simulated_data['experiment_id']
normalized_data_label.loc[sample_ids].head(10)


# ## Visualize simulated data (gene space) projected into UMAP space

# In[11]:


# UMAP embedding of original input data

# Get and save model
model = umap.UMAP(random_state=randomState).fit(normalized_data)

input_data_UMAPencoded = model.transform(normalized_data)
input_data_UMAPencoded_df = pd.DataFrame(data=input_data_UMAPencoded,
                                         index=normalized_data.index,
                                         columns=['1','2'])
# Add label
input_data_UMAPencoded_df['experiment_id'] = normalized_data_label['experiment_id']


# In[12]:


# UMAP embedding of simulated data

# Drop label column
simulated_data_numeric = simulated_data.drop(['experiment_id'], axis=1)

simulated_data_UMAPencoded = model.transform(simulated_data_numeric)
simulated_data_UMAPencoded_df = pd.DataFrame(data=simulated_data_UMAPencoded,
                                         index=simulated_data.index,
                                         columns=['1','2'])

# Add back label column
simulated_data_UMAPencoded_df['experiment_id'] = simulated_data['experiment_id']


# In[13]:


# Add label for input or simulated dataset
input_data_UMAPencoded_df['dataset'] = 'original'
simulated_data_UMAPencoded_df['dataset'] = 'simulated'

# Concatenate input and simulated dataframes together
combined_data_df = pd.concat([input_data_UMAPencoded_df, simulated_data_UMAPencoded_df])

# Plot sequentially
#backgrd_data = combined_data_df[combined_data_df['experiment_id'] == 'Not selected']
#select_data = combined_data_df[combined_data_df['experiment_id'] != 'Not selected']

# Plot
ggplot(combined_data_df, aes(x='1', y='2')) + geom_point(aes(color='experiment_id'), alpha=0.3) + facet_wrap('~dataset') + xlab('UMAP 1') + ylab('UMAP 2') + ggtitle('UMAP of original and simulated data (gene space)')
#+ xlim(3,12) \
#+ ylim(-7,10) \
#+ scale_colour_manual(values=["blue", "purple", "orange", "red", "magenta", "lightgrey"]) \


# In[14]:


# Overlay original and simulated data 
ggplot(combined_data_df, aes(x='1', y='2')) + geom_point(aes(color='dataset'), alpha=0.3) + scale_colour_manual(values=["grey", "blue"]) + xlab('UMAP 1') + ylab('UMAP 2') + ggtitle('UMAP of original and simulated data (gene space)')

# Zoomed in view

# Add label for input or simulated dataset
#input_data_UMAPencoded_df['dataset'] = 'original'
#simulated_data_UMAPencoded_df['dataset'] = 'simulated'

# Concatenate input and simulated dataframes together
#combined_data_df = pd.concat([input_data_UMAPencoded_df, simulated_data_UMAPencoded_df])

# Plot
#ggplot(combined_data_df, aes(x='1', y='2')) \
#+ geom_point(aes(color='experiment_id'), alpha=1) \
#+ facet_wrap('~dataset') \
#+ xlab('UMAP 1') \
#+ ylab('UMAP 2') \
#+ xlim(-2, 7) \
#+ ylim(-4, 7) \
#+ ggtitle('UMAP of original and simulated data (gene space)')
# ## Visualize simulated data (gene space) projected into PCA space

# In[15]:


# UMAP embedding of original input data

# Get and save model
pca = PCA(n_components=2)
pca.fit(normalized_data)

input_data_PCAencoded = pca.transform(normalized_data)
input_data_PCAencoded_df = pd.DataFrame(data=input_data_PCAencoded,
                                         index=normalized_data.index,
                                         columns=['1','2'])
# Add label
input_data_PCAencoded_df['experiment_id'] = normalized_data_label['experiment_id']


# In[16]:


# UMAP embedding of simulated data

# Drop label column
simulated_data_numeric = simulated_data.drop(['experiment_id'], axis=1)

simulated_data_PCAencoded = pca.transform(simulated_data_numeric)
simulated_data_PCAencoded_df = pd.DataFrame(data=simulated_data_PCAencoded,
                                         index=simulated_data.index,
                                         columns=['1','2'])

# Add back label column
simulated_data_PCAencoded_df['experiment_id'] = simulated_data['experiment_id']


# In[17]:


# Add label for input or simulated dataset
input_data_PCAencoded_df['dataset'] = 'original'
simulated_data_PCAencoded_df['dataset'] = 'simulated'

# Concatenate input and simulated dataframes together
combined_data_df = pd.concat([input_data_PCAencoded_df, simulated_data_PCAencoded_df])

# Plot
ggplot(combined_data_df, aes(x='1', y='2')) + geom_point(aes(color='experiment_id'), alpha=0.3) + facet_wrap('~dataset') + xlab('PCA 1') + ylab('PCA 2') + ggtitle('PCA of original and simulated data (gene space)')
#+ scale_colour_manual(values=["blue", "purple", "orange", "red", "magenta", "lightgrey"]) \


# ## Visualize simulated data (latent space) projected into UMAP space

# In[18]:


# Encode original gene expression data into latent space
data_encoded_all = loaded_model.predict_on_batch(normalized_data)
data_encoded_all_df = pd.DataFrame(data_encoded_all, index=normalized_data.index)

data_encoded_all_df.head()


# In[19]:


# Get and save model
model = umap.UMAP(random_state=randomState).fit(data_encoded_all_df)

input_data_UMAPencoded = model.transform(data_encoded_all_df)
input_data_UMAPencoded_df = pd.DataFrame(data=input_data_UMAPencoded,
                                         index=data_encoded_all_df.index,
                                         columns=['1','2'])
# Add label
input_data_UMAPencoded_df['experiment_id'] = normalized_data_label['experiment_id']


# In[20]:


# Encode simulated gene expression data into latent space
simulated_data_numeric = simulated_data.drop(['experiment_id'], axis=1)

simulated_data_encoded = loaded_model.predict_on_batch(simulated_data_numeric)
simulated_data_encoded_df = pd.DataFrame(simulated_data_encoded, index=simulated_data.index)

simulated_data_encoded_df.head()


# In[21]:


# Use same UMAP projection to plot simulated data
simulated_data_UMAPencoded = model.transform(simulated_data_encoded_df)
simulated_data_UMAPencoded_df = pd.DataFrame(data=simulated_data_UMAPencoded,
                                         index=simulated_data.index,
                                         columns=['1','2'])

# Add back label column
simulated_data_UMAPencoded_df['experiment_id'] = simulated_data['experiment_id']


# In[22]:


# Add label for input or simulated dataset
input_data_UMAPencoded_df['dataset'] = 'original'
simulated_data_UMAPencoded_df['dataset'] = 'simulated'

# Concatenate input and simulated dataframes together
combined_data_df = pd.concat([input_data_UMAPencoded_df, simulated_data_UMAPencoded_df])

# Plot
ggplot(combined_data_df, aes(x='1', y='2')) + geom_point(aes(color='experiment_id'), alpha=0.3) + facet_wrap('~dataset') + xlab('UMAP 1') + ylab('UMAP 2') + ggtitle('UMAP of original and simulated data (latent space)')
#+ xlim(-10,5) \
#+ ylim(-10,10) \
#+ scale_colour_manual(values=["blue", "purple", "orange", "red", "magenta", "lightgrey"]) \


# ## Visualize simulated data (latent space) projected into PCA space

# In[23]:


# Get and save model
pca = PCA(n_components=2)
pca.fit(data_encoded_all_df)

input_data_PCAencoded = pca.transform(data_encoded_all_df)
input_data_PCAencoded_df = pd.DataFrame(data=input_data_PCAencoded,
                                         index=data_encoded_all_df.index,
                                         columns=['1','2'])
# Add label
input_data_PCAencoded_df['experiment_id'] = normalized_data_label['experiment_id']


# In[24]:


# Use same UMAP projection to plot simulated data
simulated_data_PCAencoded = pca.transform(simulated_data_encoded_df)
simulated_data_PCAencoded_df = pd.DataFrame(data=simulated_data_PCAencoded,
                                         index=simulated_data.index,
                                         columns=['1','2'])

# Add back label column
simulated_data_PCAencoded_df['experiment_id'] = simulated_data['experiment_id']


# In[25]:


# Add label for input or simulated dataset
input_data_PCAencoded_df['dataset'] = 'original'
simulated_data_PCAencoded_df['dataset'] = 'simulated'

# Concatenate input and simulated dataframes together
combined_data_df = pd.concat([input_data_PCAencoded_df, simulated_data_PCAencoded_df])

# Plot
ggplot(combined_data_df, aes(x='1', y='2')) + geom_point(aes(color='experiment_id'), alpha=0.4) + facet_wrap('~dataset') + xlab('PCA 1') + ylab('PCA 2') + ggtitle('PCA of original and simulated data (latent space)')
#+ scale_colour_manual(values=["blue", "purple", "orange", "red", "magenta", "lightgrey"]) \

