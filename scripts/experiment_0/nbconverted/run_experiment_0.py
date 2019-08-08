#!/usr/bin/env python
# coding: utf-8

# # Main
# Run pipeline for experiment 0.
# 
# The goal of experiment 0 is to validate the approach of simulating batch effects data and using the SVCCA similarity metric to determine the affect each batch effect has on the representation of the data.

# In[1]:


import os
import ast


# In[2]:


# Load config file
config_file = os.path.join(
    os.path.abspath(os.path.join(os.getcwd(),"../..")),
    "data",
    "metadata",
    "config_exp_0.txt")

d = {}
float_params = ["learning_rate", "kappa", "epsilon_std"]
str_params = ["analysis_name", "NN_architecture"]
lst_params = ["num_batches"]
with open(config_file) as f:
    for line in f:
        (name, val) = line.split()
        if name in float_params:
            d[name] = float(val)
        elif name in str_params:
            d[name] = str(val)
        elif name in lst_params:
            d[name] = ast.literal_eval(val)
        else:
            d[name] = int(val)


# In[3]:


# Print params
print("Parameters:")
for name, val in d.items():
    print("{} = {}".format(name, val))


# In[4]:


# Training

print("Training VAE using params...")
get_ipython().run_line_magic('run', './1_train_vae.ipynb')


# In[5]:


# Simulate data

print("Simulating data...")
get_ipython().run_line_magic('run', './2_simulate_data.ipynb')


# In[6]:


# Add batch effects to simulated data

print("Adding batch effects to simulated data...")
get_ipython().run_line_magic('run', './3_add_batch_effects.ipynb')


# In[7]:


# Calculate similarity between data with different batch effects

print("Calculating similarity between representations...")
get_ipython().run_line_magic('run', './4_similarity_analysis.ipynb')

