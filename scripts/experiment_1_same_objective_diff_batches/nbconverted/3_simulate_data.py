
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
analysis_name = 'experiment_1'
NN_architecture = 'NN_2500_20'
metadata_field = 'strain'
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

# Output
simulated_data_file = os.path.join(
    base_dir,
    "data",
    "simulated",
    analysis_name,
    "simulated_data.txt")

simulated_data_mapping_file = os.path.join(
    base_dir,
    "data",
    "metadata",
    "mapping_simulated_data.txt")

umap_model_file = os.path.join(
    base_dir,
    "models",  
    NN_architecture,
    "umap_model.pkl")


# In[5]:


# Read in VAE models
loaded_model = load_model(model_encoder_file)
loaded_decode_model = load_model(model_decoder_file)

loaded_model.load_weights(weights_encoder_file)
loaded_decode_model.load_weights(weights_decoder_file)


# In[6]:


# Read in UMAP model
infile = open(umap_model_file, 'rb')
model = pickle.load(infile)
infile.close()


# In[7]:


# Read data
normalized_data = pd.read_table(
    normalized_data_file,
    header=0,
    sep='\t',
    index_col=0).T

print(normalized_data.shape)
normalized_data.head(10)


# In[8]:


# Read encoded data
encoded_data = pd.read_table(
    encoded_data_file,
    header=0,
    sep='\t',
    index_col=0)

print(encoded_data.shape)
encoded_data.head(10)


# In[9]:


# Read in metadata
metadata = pd.read_table(
    mapping_file, 
    header=0, 
    sep='\t', 
    index_col=0)

metadata.head(10)


# In[10]:


# Replace NaN with string "NA"
metadata[metadata_field] = metadata[metadata_field].fillna('NA')


# In[11]:


# Get possible values in metadata field
#grps = list(metadata[metadata_field].unique())  #Want a specific signal, test if this works
grps = ['PAO1', 'PA14']
print(grps)


# ## Simulate data
# 
# Generate new simulated data by sampling from the distribution of latent space features.  In other words, for each latent space feature get the mean and standard deviation.  Then we can generate a new sample by sampling from a distribution with this mean and standard deviation.

# In[12]:


# Encode into latent space
data_encoded = loaded_model.predict_on_batch(normalized_data)
data_encoded_df = pd.DataFrame(data_encoded, index=normalized_data.index)


# In[13]:


# Merge encoded gene expression data and metadata
data_encoded_labeled = encoded_data.merge(
    metadata,
    left_index=True, 
    right_index=True, 
    how='inner')

print(data_encoded_labeled.shape)
data_encoded_labeled.head(5)


# In[14]:


# Init variables
num_samples_per_grp = int(num_simulated_samples/len(grps))
latent_dim = data_encoded_df.shape[1]
new_data = pd.DataFrame(columns=normalized_data.columns)
new_data_encoded = pd.DataFrame(columns=encoded_data.columns)

# Get mean and standard deviation for each group per encoded feature
for g in range(len(grps)):
    grp_name = grps[g]
    data_labeled_grped = data_encoded_labeled[data_encoded_labeled[metadata_field]== grps[g]]
    
    print('simulating data for {}...'.format(grp_name))
    
    # Calculate mean and stdev
    encoded_means = data_labeled_grped.mean(axis=0)
    encoded_stds = data_labeled_grped.std(axis=0)

    # Generate samples 
    new_data_tmp = np.zeros([num_samples_per_grp,latent_dim])
    for j in range(latent_dim):
        
        # Use mean and std for feature
        new_data_tmp[:,j] = np.random.normal(encoded_means[j], encoded_stds[j], num_samples_per_grp) 
        
        # Use standard normal
        #new_data[:,j] = np.random.normal(0, 1, num_simulated_samples)
        
    new_data_tmp_df = pd.DataFrame(data=new_data_tmp, columns=encoded_data.columns)
    new_data_encoded = new_data_encoded.append(new_data_tmp_df, ignore_index=True)
    

    # Decode N samples
    new_data_decoded = loaded_decode_model.predict_on_batch(new_data_tmp_df)
    new_data_decoded_df = pd.DataFrame(data=new_data_decoded, columns=normalized_data.columns)
    
    new_data = new_data.append(new_data_decoded_df, ignore_index=True)

print(new_data.shape)
new_data.head(10)


# In[15]:


# Create labels for new data
grps_series = pd.Series(grps)
new_metadata = pd.DataFrame(grps_series.repeat(num_samples_per_grp), columns=['metadata'])
new_metadata.index = new_data.index


# In[16]:


# Merge gene expression data and metadata
new_data_labeled = new_data.merge(
    new_metadata,
    left_index=True, 
    right_index=True, 
    how='inner')

print(new_data_labeled.shape)
new_data_labeled.head(5)


# ## Plot simulated input data using UMAP
# 
# Note: we will use the same UMAP mapping for the input and simulated data to ensure they are plotted on the same space.

# In[17]:


# UMAP embedding
simulated_data_UMAP = model.transform(new_data_labeled.iloc[:,:-1])
simulated_data_UMAP_df = pd.DataFrame(data=simulated_data_UMAP,
                                         index=new_data_labeled.index,
                                         columns=['1','2'])
simulated_data_UMAP_df['metadata'] = list(new_data_labeled['metadata'])

g = ggplot(aes(x='1',y='2', color='metadata'), data=simulated_data_UMAP_df) +             geom_point(alpha=0.7) +             scale_color_brewer(type='qual', palette='Set1') +             ggtitle("Simulated data")

print(g)


# In[18]:


# Output
new_data.to_csv(simulated_data_file, sep='\t')
new_metadata.to_csv(simulated_data_mapping_file, sep='\t')

