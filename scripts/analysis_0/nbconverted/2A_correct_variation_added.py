#!/usr/bin/env python
# coding: utf-8

# # Correct for the technical variation added
# 
# The goal of this notebook is to try to correct for the technical variation that was added by each experiment
# 
# The approach is to,
# 1. Import the simulated data representing varying experiments
# 2. Use [removeBatchEffect](https://rdrr.io/bioc/limma/man/removeBatchEffect.html) package from the limma library in R.
# 3. Calculate the similarity between the dataset with a single experiment and the dataset corrected for the variation introduced by having some number of experiments added.

# In[18]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('load_ext', 'rpy2.ipython')

import rpy2
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
limma = importr('limma')

import os
import sys
import glob
import pickle
import pandas as pd
import numpy as np
from plotnine import (ggplot, 
                      labs,  
                      geom_line, 
                      aes, 
                      ggsave, 
                      theme_bw,
                      theme,
                      element_text,
                      element_rect,
                      element_line)
import warnings
warnings.filterwarnings(action='ignore')

sys.path.append("../")
from functions import generate_data
from functions import similarity_metric

from numpy.random import seed
randomState = 123
seed(randomState)


# In[2]:


get_ipython().run_cell_magic('R', '', '# Run once to install needed R packages\n#install.packages(c("devtools"))\n#source("http://www.bioconductor.org/biocLite.R")\n#biocLite(c("limma"))\nlibrary(limma)')


# In[3]:


# User parameters
NN_architecture = 'NN_2500_30'
analysis_name = 'analysis_0'
num_simulated_samples = 6000
lst_num_experiments = [1,2,5,10,20,50,100,500,1000,2000,3000,6000]
use_pca = True
num_PCs = 10


# In[4]:


# Input files
base_dir = os.path.abspath(os.path.join(os.getcwd(),"../.."))    # base dir on repo
local_dir = "/home/alexandra/Documents/"                         # base dir on local machine for data storage

# Simulated data file 
simulated_data_file = os.path.join(
    local_dir,
    "Data",
    "Batch_effects",
    "simulated",
    analysis_name,
    "simulated_data.txt.xz")


# In[19]:


# Output file
svcca_file = os.path.join(
    local_dir,
    "Data",
    "Batch_effects",
    "output",
    "analysis_0_svcca_correction.png")

svcca_blk_file = os.path.join(
    local_dir,
    "Data",
    "Batch_effects",
    "output",
    "analysis_0_svcca_correction_blk.png")

similarity_corrected_file = os.path.join(
    local_dir,
    "Data",
    "Batch_effects",
    "output",
    "analysis_0_similarity_corrected.pickle")


# ### Correct for added variation

# In[6]:


for i in lst_num_experiments:
    print('Correcting for {} experiments..'.format(i))

    # Simulated data with experiments added
    experiment_file = os.path.join(
        local_dir,
        "Data",
        "Batch_effects",
        "experiment_simulated",
        analysis_name,
        "Experiment_" + str(i) + ".txt.xz")
    
    # Read in data
    # data transposed to form gene x sample for R package
    experiment_data = pd.read_table(
        experiment_file,
        header=0,
        index_col=0,
        sep='\t').T
    
    # Experimental ids
    experiment_map_file = os.path.join(
        local_dir,
        "Data",
        "Batch_effects",
        "experiment_simulated",
        analysis_name,
        "Experiment_map_" + str(i) + ".txt.xz")
    
    # Read in map
    experiment_map = pd.read_table(
        experiment_map_file,
        header=0,
        index_col=0,
        sep='\t')['experiment']
    
    if i == 1:
        corrected_experiment_data_df = experiment_data.copy()
    
    else:    
        # Correct for technical variation
        corrected_experiment_data = limma.removeBatchEffect(experiment_data, batch=experiment_map)

        # Convert R object to pandas df
        corrected_experiment_data_df = pandas2ri.ri2py_dataframe(corrected_experiment_data)
    
    # Write out corrected files
    experiment_corrected_file = os.path.join(
        local_dir,
        "Data",
        "Batch_effects",
        "experiment_simulated",
        analysis_name,
        "Experiment_corrected_" + str(i) + ".txt.xz")
    
    corrected_experiment_data_df.to_csv(
        experiment_corrected_file, float_format='%.3f', sep='\t', compression='xz')


# ### Calculate similarity

# In[7]:


# Permuted simulated data file 
permuted_simulated_data_file = os.path.join(
    local_dir,
    "Data",
    "Batch_effects",
    "simulated",
    analysis_name,
    "permuted_simulated_data.txt.xz")


# In[8]:


# Calculate similarity
batch_scores, permuted_score = similarity_metric.sim_svcca(simulated_data_file,
                                                           permuted_simulated_data_file,
                                                           "Experiment_corrected",
                                                           lst_num_experiments,
                                                           use_pca,
                                                           num_PCs,
                                                           local_dir,
                                                           analysis_name)


# In[9]:


# Convert similarity scores to pandas dataframe
similarity_score_df = pd.DataFrame(data={'score': batch_scores},
                                     index=lst_num_experiments,
                                    columns=['score'])
similarity_score_df.index.name = 'number of experiments'
similarity_score_df


# In[10]:


print("Similarity between input vs permuted data is {}".format(permuted_score))


# In[16]:


# Plot
threshold = pd.DataFrame(
    pd.np.tile(
        permuted_score,
        (len(lst_num_experiments), 1)),
    index=lst_num_experiments,
    columns=['score'])

g = ggplot(similarity_score_df, aes(x=lst_num_experiments, y='score'))     + geom_line()     + geom_line(aes(x=lst_num_experiments, y='score'), threshold, linetype='dashed')     + labs(x = "Number of Experiments", 
           y = "Similarity score (SVCCA)", 
           title = "Similarity after correcting for experiment variation") \
    + theme_bw() \
    + theme(plot_title=element_text(weight='bold'))

print(g)
ggsave(plot=g, filename=svcca_file, dpi=300)


# In[17]:


# Plot - black
threshold = pd.DataFrame(
    pd.np.tile(
        permuted_score,
        (len(lst_num_experiments), 1)),
    index=lst_num_experiments,
    columns=['score'])

g = ggplot(similarity_score_df, aes(x=lst_num_experiments, y='score'))     + geom_line(colour="white")     + geom_line(aes(x=lst_num_experiments, y='score'), threshold, colour="white", linetype='dashed')     + labs(x = "Number of Experiments", 
           y = "Similarity score (SVCCA)", 
           title = "Similarity after correcting for experiment variation") \
    + theme(plot_title=element_text(weight='bold', colour="white"),
            plot_background=element_rect(fill="black"),
            panel_background=element_rect(fill="black"),
            axis_title_x=element_text(colour="white"),
            axis_title_y=element_text(colour="white"),
            axis_line=element_line(color="white"),
            axis_text=element_text(color="white")
           )

print(g)
ggsave(plot=g, filename=svcca_blk_file, dpi=300)


# In[20]:


# Pickle similarity scores to overlay uncorrected and corrected svcca curves
similarity_score_df.to_pickle(similarity_corrected_file)

