'''
Author: Alexandra Lee
Date Created: 11 November 2019

Scripts to run all scripts in one function in order to run the entire pipeline multiple times in parallel
'''
import os
import sys
import glob
import pickle
import pandas as pd
import numpy as np

import rpy2
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
limma = importr('limma')

import warnings
warnings.filterwarnings(action='ignore')

sys.path.append("../")
from functions import generate_data_parallel
from functions import similarity_metric_parallel


def simple_simulation_experiment_uncorrected():

  # Parameters
  NN_architecture = 'NN_2500_30'
  analysis_name = 'analysis_0'
  # num_simulated_samples = 6000
  # lst_num_experiments = [1, 2, 5, 10, 20]
  num_simulated_samples = 6000
  lst_num_experiments = [1, 2, 5, 10, 20,
                         50, 100, 500, 1000, 2000, 3000, 6000]
  corrected = False
  use_pca = True
  num_PCs = 10

  # Input files
  base_dir = os.path.abspath(
      os.path.join(
          os.getcwd(), "../.."))    # base dir on repo

  local_dir = "/home/alexandra/Documents/"

  normalized_data_file = os.path.join(
      base_dir,
      "data",
      "input",
      "train_set_normalized.pcl")

  # Main

  # Generate simulated data
  simulated_data = generate_data_parallel.simulate_data(normalized_data_file,
                                                        NN_architecture,
                                                        analysis_name,
                                                        num_simulated_samples
                                                        )

  # Permute simulated data to be used as a negative control
  permuted_data = generate_data_parallel.permute_data(simulated_data,
                                                      local_dir,
                                                      analysis_name)

  # Add technical variation
  lst_compendia, lst_compendia_labels = generate_data_parallel.add_experiments(simulated_data,
                                                                               lst_num_experiments,
                                                                               local_dir,
                                                                               analysis_name)

  # Calculate similarity between compendium and compendium + noise
  batch_scores, permuted_score = similarity_metric_parallel.sim_svcca(simulated_data,
                                                                      permuted_data,
                                                                      lst_compendia,
                                                                      corrected,
                                                                      lst_num_experiments,
                                                                      use_pca,
                                                                      num_PCs,
                                                                      local_dir,
                                                                      analysis_name)

  # Convert similarity scores to pandas dataframe
  similarity_score_df = pd.DataFrame(data={'score': batch_scores},
                                     index=lst_num_experiments,
                                     columns=['score'])

  similarity_score_df.index.name = 'number of experiments'
  similarity_score_df

  # Return similarity scores and permuted score
  return permuted_score, similarity_score_df, lst_compendia


def simple_simulation_experiment_corrected(run):
  # Parameters
  NN_architecture = 'NN_2500_30'
  analysis_name = 'analysis_0'
  file_prefix = 'Experiment_corrected'
  num_simulated_samples = 6000
  lst_num_experiments = [1, 2, 5, 10, 20,
                         50, 100, 500, 1000, 2000, 3000, 6000]
  corrected = True
  use_pca = True
  num_PCs = 10

  # Input files
  base_dir = os.path.abspath(
      os.path.join(
          os.getcwd(), "../.."))    # base dir on repo

  local_dir = "/home/alexandra/Documents/"

  normalized_data_file = os.path.join(
      base_dir,
      "data",
      "input",
      "train_set_normalized.pcl")

  # Main

  # Generate simulated data
  simulated_data = generate_data_parallel.simulate_data(normalized_data_file,
                                                        NN_architecture,
                                                        analysis_name,
                                                        num_simulated_samples
                                                        )

  # Permute simulated data to be used as a negative control
  permuted_data = generate_data_parallel.permute_data(simulated_data,
                                                      local_dir,
                                                      analysis_name)

  # Add technical variation
  generate_data_parallel.add_experiments_io(simulated_data,
                                            lst_num_experiments,
                                            run,
                                            local_dir,
                                            analysis_name)

  # Remove technical variation
  generate_data_parallel.apply_correction_io(local_dir,
                                             run,
                                             analysis_name,
                                             lst_num_experiments)

  # Calculate similarity between compendium and compendium + noise
  batch_scores, permuted_score = similarity_metric_parallel.sim_svcca_io(simulated_data,
                                                                         permuted_data,
                                                                         corrected,
                                                                         file_prefix,
                                                                         run,
                                                                         lst_num_experiments,
                                                                         use_pca,
                                                                         num_PCs,
                                                                         local_dir,
                                                                         analysis_name)

  # batch_scores, permuted_score

  # Convert similarity scores to pandas dataframe
  similarity_score_df = pd.DataFrame(data={'score': batch_scores},
                                     index=lst_num_experiments,
                                     columns=['score'])

  similarity_score_df.index.name = 'number of experiments'
  similarity_score_df

  # Return similarity scores and permuted score
  return permuted_score, similarity_score_df
