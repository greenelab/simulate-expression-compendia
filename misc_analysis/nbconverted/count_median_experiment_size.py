
# coding: utf-8

# # Number of experiments
# 
# This notebook counts the number of unique *experiments* within the recount2 and *P. aeruginosa* compendia.  These statistics are used in the corresponding manuscript

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import sys
import os
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings(action='ignore')


# In[2]:


# Parameters
dataset_name = "Human_analysis"


# In[3]:


base_dir = os.path.abspath(os.path.join(os.getcwd(),"../"))    # base dir on repo

if dataset_name == "Pseudomonas_analysis":    
    metadata_file = os.path.join(
        base_dir,
        dataset_name,    
        "data",
        "metadata",
        "sample_annotations.tsv")
else:
    metadata_file = os.path.join(
        base_dir,
        dataset_name,    
        "data",
        "metadata",
        "recount2_metadata.tsv")


# In[6]:


metadata = pd.read_table(
    metadata_file,
    header=0,
    sep='\t',
    index_col=0)

print(metadata.shape)
metadata.head()


# In[5]:


metadata.index.value_counts().median()

