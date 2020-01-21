
# coding: utf-8

# ## Import data and normalize

# In[1]:


import os
import pandas as pd
from sklearn import preprocessing
import umap
from plotnine import (ggplot, 
                      geom_point,
                      labs,
                      aes)
from numpy.random import seed
randomState = 123
seed(randomState)

import warnings
warnings.filterwarnings(action='ignore')


# In[2]:


# User parameters
dataset_name = "Human_analysis"


# In[3]:


# Load arguments
base_dir = os.path.abspath(os.path.join(os.getcwd(),"../.."))

rpkm_data_file = os.path.join(
    base_dir,
    dataset_name,
    "data",
    "input",
    "recount2_gene_RPKM_data.tsv")


# In[4]:


# Output file
out_data_file = os.path.join(
    base_dir,
    dataset_name,
    "data",
    "input",
    "recount2_gene_normalized_data.tsv.xz")


# In[5]:


# Read data
rpkm_data = pd.read_table(
    rpkm_data_file,
    header=0,
    sep='\t',
    index_col=0)

rpkm_data.head()
print(rpkm_data.shape)


# In[6]:


# 0-1 normalize per gene
rnaseq_scaled_df = preprocessing.MinMaxScaler().fit_transform(rpkm_data)
rnaseq_scaled_df = pd.DataFrame(rnaseq_scaled_df,
                                columns=rpkm_data.columns,
                                index=rpkm_data.index).T

rnaseq_scaled_df.head()


# In[7]:


# UMAP embedding of original input data
model = umap.UMAP(random_state=randomState).fit(rnaseq_scaled_df.T)

input_data_UMAPencoded = model.transform(rnaseq_scaled_df.T)
input_data_UMAPencoded_df = pd.DataFrame(data=input_data_UMAPencoded,
                                         index=rnaseq_scaled_df.T.index,
                                         columns=['1','2'])


g_input = ggplot(input_data_UMAPencoded_df, aes(x='1',y='2'))     + geom_point(alpha=0.3)     + labs(x = "UMAP 1", y = "UMAP 2", title = "Input data") 
print(g_input)


# In[8]:


# Save scaled data
rnaseq_scaled_df.to_csv(out_data_file, sep='\t', compression='xz')

