
# coding: utf-8

# # Visualize data
# 
# In order to start making interpretations we will generate two visualizations of our data
# 
# First, we will verify that the simulated dataset is a good representation of our original input dataset by visually comparing the structures in the two datasets projected onto UMAP space.
# 
# Second, we will plot the PCA project data after adding batch effects to examine how the batch effects shift the data

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import os
import ast
import pandas as pd
import numpy as np
import random
import glob
from plotnine import ggplot, ggtitle, xlab, ylab, geom_point, aes, facet_wrap
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
analysis_name = 'analysis_0'
num_dims=5000
num_simulated_samples = 6000
lst_num_experiments = [1,2,5,10,20,50,100,500,1000,2000,3000,6000]


# In[3]:


# Load data
base_dir = os.path.abspath(os.path.join(os.getcwd(),"../.."))
NN_dir = base_dir + "/models/" + NN_architecture

normalized_data_file = os.path.join(
    base_dir,
    "data",
    "input",
    "train_set_normalized.pcl")

model_encoder_file = glob.glob(os.path.join(
    NN_dir,
    "*_encoder_model.h5"))[0]

weights_encoder_file = glob.glob(os.path.join(
    NN_dir,
    "*_encoder_weights.h5"))[0]

model_decoder_file = glob.glob(os.path.join(
    NN_dir,
    "*_decoder_model.h5"))[0]

weights_decoder_file = glob.glob(os.path.join(
    NN_dir,
    "*_decoder_weights.h5"))[0]

experiment_dir = os.path.join(
    base_dir,
    "data",
    "experiment_simulated",
    analysis_name)

simulated_data_file = os.path.join(
    base_dir,
    "data",
    "simulated",
    analysis_name,
    "simulated_data.txt.xz")

permuted_simulated_data_file = os.path.join(
    base_dir,
    "data",
    "simulated",
    analysis_name,
    "permuted_simulated_data.txt.xz")


# ## Visualize simulated data in UMAP space

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


# UMAP embedding of original input data

# Get and save model
model = umap.UMAP(random_state=randomState).fit(normalized_data)

input_data_UMAPencoded = model.transform(normalized_data)
input_data_UMAPencoded_df = pd.DataFrame(data=input_data_UMAPencoded,
                                         index=normalized_data.index,
                                         columns=['1','2'])


ggplot(input_data_UMAPencoded_df, aes(x='1',y='2'))     + geom_point(alpha=0.5)     + ggtitle('Input data')


# In[9]:


# UMAP embedding of simulated data
## When dimensions are the same then can use the same UMAP projection, but for now it is different
simulated_data_UMAPencoded = umap.UMAP(random_state=randomState).fit_transform(simulated_data)
simulated_data_UMAPencoded_df = pd.DataFrame(data=simulated_data_UMAPencoded,
                                         index=simulated_data.index,
                                         columns=['1','2'])


ggplot(simulated_data_UMAPencoded_df, aes(x='1',y='2'))     + geom_point(alpha=0.5)     + ggtitle("Simulated data")


# In[10]:


## Side by side original input vs simulated data

# Add label for input or simulated dataset
#input_data_UMAPencoded_df['dataset'] = 'original'
#simulated_data_UMAPencoded_df['dataset'] = 'simulated'

# Concatenate input and simulated dataframes together
#combined_data_df = pd.concat([input_data_UMAPencoded_df, simulated_data_UMAPencoded_df])

# Plot
#ggplot(combined_data_df, aes(x='1', y='2')) \
#+ geom_point(alpha=0.3) \
#+ facet_wrap('~dataset') \
#+ xlab('UMAP 1') \
#+ ylab('UMAP 2') \
#+ ggtitle('UMAP of original and simulated data')


# ## Visualize effects of multiple experiments in PCA space

# In[11]:


get_ipython().run_cell_magic('time', '', '\nall_data_df = pd.DataFrame()\n\n# Get batch 1 data\nexperiment_1_file = os.path.join(\n    experiment_dir,\n    "Experiment_1.txt.xz")\n\nexperiment_1 = pd.read_table(\n    experiment_1_file,\n    header=0,\n    index_col=0,\n    sep=\'\\t\')\n\n\nfor i in lst_num_experiments:\n    print(\'Plotting PCA of 1 experiment vs {} experiments...\'.format(i))\n    \n    # Simulated data with all samples in a single batch\n    original_data_df =  experiment_1.copy()\n    \n    # Add grouping column for plotting\n    original_data_df[\'group\'] = \'experiment_1\'\n    \n    # Get data with additional batch effects added\n    experiment_other_file = os.path.join(\n        experiment_dir,\n        "Experiment_"+str(i)+".txt.xz")\n\n    experiment_other = pd.read_table(\n        experiment_other_file,\n        header=0,\n        index_col=0,\n        sep=\'\\t\')\n    \n    # Simulated data with i batch effects\n    experiment_data_df =  experiment_other\n    \n    # Add grouping column for plotting\n    experiment_data_df[\'group\'] = "experiment_{}".format(i)\n    \n    # Concatenate datasets together\n    combined_data_df = pd.concat([original_data_df, experiment_data_df])\n    \n    # PCA projection\n    pca = PCA(n_components=2)\n\n    # Encode expression data into 2D PCA space\n    combined_data_numeric_df = combined_data_df.drop([\'group\'], axis=1)\n    combined_data_PCAencoded = pca.fit_transform(combined_data_numeric_df)\n\n\n    combined_data_PCAencoded_df = pd.DataFrame(combined_data_PCAencoded,\n                                               index=combined_data_df.index,\n                                               columns=[\'PC1\', \'PC2\']\n                                              )\n    \n    # Add back in batch labels (i.e. labels = "batch_"<how many batch effects were added>)\n    combined_data_PCAencoded_df[\'group\'] = combined_data_df[\'group\']\n    \n    # Add column that designates which batch effect comparision (i.e. comparison of 1 batch vs 5 batches\n    # is represented by label = 5)\n    combined_data_PCAencoded_df[\'num_experiments\'] = str(i)\n    \n    # Concatenate ALL comparisons\n    all_data_df = pd.concat([all_data_df, combined_data_PCAencoded_df])\n    \n    \n    # Plot individual comparisons\n    print(ggplot(combined_data_PCAencoded_df, aes(x=\'PC1\', y=\'PC2\')) \\\n          + geom_point(aes(color=\'group\'), alpha=0.4) \\\n          + xlab(\'PC1\') \\\n          + ylab(\'PC2\') \\\n          + ggtitle(\'Experiment 1 and Experiment {}\'.format(i))\n         )')


# In[12]:


# Plot all comparisons in one figure
ggplot(all_data_df, aes(x='PC1', y='PC2')) + geom_point(aes(color='group'), alpha=0.3) + facet_wrap('~num_experiments') + xlab('PC1') + ylab('PC2') + ggtitle('PCA of experiment 1 vs experiment x')


# ## Permuted dataset (Negative control)
# 
# As a negative control we will permute the values within a sample, across genes in order to disrupt the gene expression structure.

# In[13]:


# Read in permuated data
shuffled_simulated_data = pd.read_table(
    permuted_simulated_data_file,
    header=0,
    index_col=0,
    sep='\t')


# In[14]:


# Label samples with label = perumuted
shuffled_simulated_data['group'] = "permuted"

# Concatenate original simulated data and shuffled simulated data
input_vs_permuted_df = pd.concat([original_data_df, shuffled_simulated_data])


input_vs_permuted = input_vs_permuted_df.drop(['group'], axis=1)
shuffled_data_PCAencoded = pca.fit_transform(input_vs_permuted)


shuffled_data_PCAencoded_df = pd.DataFrame(shuffled_data_PCAencoded,
                                           index=input_vs_permuted_df.index,
                                           columns=['PC1', 'PC2']
                                          )

# Add back in batch labels (i.e. labels = "batch_"<how many batch effects were added>)
shuffled_data_PCAencoded_df['group'] = input_vs_permuted_df['group']


# In[15]:


# Plot permuted data
print(ggplot(shuffled_data_PCAencoded_df, aes(x='PC1', y='PC2'))       + geom_point(aes(color='group'), alpha=0.4)       + xlab('PC1')       + ylab('PC2')       + ggtitle('Simulated vs Permuted')
     )

