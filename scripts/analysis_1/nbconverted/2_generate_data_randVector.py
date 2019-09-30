
# coding: utf-8

# # Generate data and calculate similarity
# 
# The goal of this notebook is to determine how much of the structure in the original dataset (single experiment) is retained after adding some number of experiments.
# 
# The approach is to,
# 1. Generates simulated data by sampling from a trained VAE model.  Simulate ```num_simulated_samples```
# 
#     Here we are trying to simulate data from a single experiment type.  

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import os
import sys
import glob
import pandas as pd
import numpy as np
import random
from plotnine import ggplot, ggtitle, xlab, ylab, geom_point, geom_line, aes
from keras.models import load_model
from sklearn.metrics.pairwise import euclidean_distances

import warnings
warnings.filterwarnings(action='ignore')

sys.path.append("../")
from functions import generate_data
from functions import similarity_metric

from numpy.random import seed
randomState = 123
seed(randomState)


# In[2]:


# User parameters
NN_architecture = 'NN_2500_30'
analysis_name = 'analysis_1'
num_simulated_samples = 100
experiment_id = 'E-GEOD-24036'
color_by_field = 'genotype'


# In[3]:


# Input files

# base dir on repo
base_dir = os.path.abspath(os.path.join(os.getcwd(),"../..")) 

# base dir on local machine for data storage
# os.makedirs doesn't recognize `~`
local_dir = local_dir = os.path.abspath(os.path.join(os.getcwd(), "../../../..")) 

NN_dir = base_dir + "/models/" + NN_architecture
latent_dim = NN_architecture.split('_')[-1]

mapping_file = os.path.join(
    base_dir,
    "data",
    "metadata",
    "sample_annotations.tsv")

normalized_data_file = os.path.join(
    base_dir,
    "data",
    "input",
    "train_set_normalized.pcl")

model_encoder_file = glob.glob(os.path.join(
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


# In[4]:


# Output file
new_dir = os.path.join(local_dir, "Data", "Batch_effects", "simulated")

analysis_dir = os.path.join(new_dir, analysis_name)

if os.path.exists(analysis_dir):
    print('Directory already exists: \n {}'.format(analysis_dir))
else:
    print('Creating new directory: \n {}'.format(analysis_dir))
os.makedirs(analysis_dir, exist_ok=True)

    
simulated_data_file = os.path.join(
    local_dir,
    "Data",
    "Batch_effects",
    "simulated",
    analysis_name,
    "simulated_data.txt.xz")


# In[5]:


# Load saved models
loaded_model = load_model(model_encoder_file)
loaded_decode_model = load_model(model_decoder_file)

loaded_model.load_weights(weights_encoder_file)
loaded_decode_model.load_weights(weights_decoder_file)


# ### Get experiment ids

# In[6]:


# Read in metadata
metadata = pd.read_table(
    mapping_file, 
    header=0, 
    sep='\t', 
    index_col=0)

metadata.head()


# In[7]:


map_experiment_sample = metadata[['sample_name', 'ml_data_source']]
map_experiment_sample.head()


# In[8]:


experiment_ids = np.unique(np.array(map_experiment_sample.index))
print("There are {} experiments in the compendium".format(len(experiment_ids)))


# ### Set user selected experiment id or randomly select experiment id

# In[9]:


if not experiment_id:
    selected_experiment_id = np.random.RandomState(randomState).choice(experiment_ids)
else:
    selected_experiment_id = experiment_id


# ### Get samples belonging to selected experiment

# In[10]:


selected_metadata = metadata.loc[selected_experiment_id]
selected_metadata.head()


# In[11]:


print("There are {} samples in experiment {}".format(selected_metadata.shape[0], selected_experiment_id))


# In[12]:


sample_ids = list(selected_metadata['ml_data_source'])
sample_ids


# In[13]:


normalized_data = pd.read_table(
    normalized_data_file,
    header=0,
    sep='\t',
    index_col=0).T

normalized_data.head()


# In[14]:


# Check that samples are in gene expression dataset otherwise exit
if any(x in list(normalized_data.index) for x in sample_ids):
    print('found')
else:
    print('not found')


# In[15]:


selected_data_df = normalized_data.loc[sample_ids]

selected_data_df.head()


# ### Embed samples into latent space using trained VAE

# In[16]:


# Encode into latent space
data_encoded = loaded_model.predict_on_batch(selected_data_df)
data_encoded_df = pd.DataFrame(data_encoded, index=selected_data_df.index)

data_encoded_df.head()


# ### Generate simulated gene expression data 
# 
# Shift samples in a different direction while maintaining the distance between samples.
# 
# The below implementation is considering each pair of closest points simulating a new sample by stepping in a random direction but preserving the distance between those two point.  Thus, by design, this implementation is preserving local structure of the experiment but not necessarily the global strucuture.

# In[17]:


randomState = 123
seed(randomState)
simulated_data_encoded_df = data_encoded_df + np.random.RandomState(randomState).normal(0, 1.0, int(latent_dim))

simulated_data_encoded_df.head()


# ### Decode simulated data into gene space

# In[18]:


# Decode simulated data into raw gene space

simulated_data_decoded = loaded_decode_model.predict_on_batch(simulated_data_encoded_df)
simulated_data_decoded_df = pd.DataFrame(simulated_data_decoded, index=simulated_data_encoded_df.index)

simulated_data_decoded_df['color_by'] = list(selected_metadata[color_by_field])
simulated_data_decoded_df.head()


# In[19]:


# Save simulated data
simulated_data_decoded_df.to_csv(simulated_data_file, float_format='%.3f', sep='\t', compression='xz')

