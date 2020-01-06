
# coding: utf-8

# In[14]:


import os
import sys
import glob
import pandas as pd
import numpy as np
import random

local_dir = local_dir = os.path.abspath(os.path.join(os.getcwd(), "../../../..")) 

normalized_data_file = os.path.join(
    local_dir,
    "Data",
    "Batch_effects",
    "input",
    "recount2_gene_normalized_data.tsv")


# In[16]:


# Read in metadata
data = pd.read_table(
    normalized_data_file, 
    header=0, 
    sep='\t', 
    index_col=0).T

data.head()


# In[17]:


data.index

