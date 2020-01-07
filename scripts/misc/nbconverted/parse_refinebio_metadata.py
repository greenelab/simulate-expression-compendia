
# coding: utf-8

# In[1]:


import pandas as pd
from statistics import median
import requests


# In[2]:


response = requests.get("http://api.refine.bio/v1/experiments/", params={'limit': 25}).json()
experiments_ls = response['results']
experiments_ls


# ### Query and parse metadata
# 
# 1. Query refine.bio api for experiment metadata
# 2. For each organism get experiment and sample counts

# In[3]:


# Get first 25 experiments
response = requests.get("http://api.refine.bio/v1/experiments/", params={'limit': 25}).json()
experiments_ls = response['results']
count = response['count']

# Initialize dictionary
num_samples_per_organism = {}

while count != 0: ### CHECK CONDITION HERE
    for experiment in experiments_ls:
        organism_name_raw = experiment['organisms']

        if len(organism_name_raw) == 1:
            samples_ls = experiment['processed_samples']
            num_samples = len(samples_ls)
            
            # Some experiments have 'processed_samples' = []
            # Not clear why this is the case
            if num_samples > 0:
                organism_name = organism_name_raw[0]
                experiment_id = experiment['id']
            
                print(experiment_id, organism_name, num_samples)

                if organism_name not in num_samples_per_organism.keys():
                    num_samples_per_organism[organism_name] = list()
                    num_samples_per_organism[organism_name].append(num_samples)

                else:
                    num_samples_per_organism[organism_name].append(num_samples)
        
    # Get next set of 25 experiments        
    url = response['next']
    if url == None:
        break
    else:
        response = requests.get(url, params={'limit': 25}).json()
        experiments_ls = response['results']        
        count -= 25


# In[4]:


num_samples_per_organism


# In[5]:


# Get the number of experiments per organism
num_experiments_per_organism = {}

for name in list(num_samples_per_organism.keys()):
    num_experiments_per_organism[name] = len(num_samples_per_organism[name])

num_experiments_per_organism


# In[48]:


# Get median number of samples per organism
median_samples_per_organism = {}

for name in list(num_samples_per_organism.keys()):
    median_samples_per_organism[name] = median(num_samples_per_organism[name])

median_samples_per_organism


# In[36]:


# Get total number of samples per organism
total_samples_per_organism = {}

for name in list(num_samples_per_organism.keys()):
    total_samples_per_organism[name] = sum(num_samples_per_organism[name])

total_samples_per_organism


# ### Create summary table

# In[50]:


# Convert dictionaries to dataframes to keep track of associations between organism name and counts
experiments_df = pd.DataFrame.from_dict(num_experiments_per_organism, orient='index')
median_df = pd.DataFrame.from_dict(median_samples_per_organism, orient='index')
total_df = pd.DataFrame.from_dict(total_samples_per_organism, orient='index')


# In[51]:


experiments_df.head()


# In[52]:


median_df.head()


# In[54]:


# Merge dataframes
tmp = experiments_df.merge(median_df, left_index=True, right_index=True)
table_df = tmp.merge(total_df, left_index=True, right_index=True)

table_df.head()


# In[56]:


# Format table
table_df.columns = ["No. experiments", "Median no. samples", "Total no. samples"]
table_df = table_df.sort_values(by=["No. experiments"], ascending=False)
print(sum(table_df["Total no. samples"]))
table_df.head(10)

