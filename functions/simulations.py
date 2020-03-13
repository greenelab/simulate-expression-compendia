'''
Author: Alexandra Lee
Date Created: 11 November 2019

Scripts to run simulation multiple times in parallel
'''
import os
import sys
import glob
import pickle
import pandas as pd
import numpy as np
import gc

import warnings
warnings.filterwarnings(action='ignore')

from functions import generate_data_parallel
from functions import similarity_metric_parallel


def sample_level_simulation(run,
                            NN_architecture,
                            dataset_name,
                            analysis_name,
                            num_simulated_samples,
                            lst_num_experiments,
                            corrected,
                            correction_method,
                            use_pca,
                            num_PCs,
                            file_prefix,
                            input_file,
                            local_dir,
                            base_dir):
  '''
    This function performs runs series of scripts that performs the following steps:
    1. Simulate gene expression data, ignorning the sample-experiment relationship
    2. Add varying numbers of technical variation
    3. Compare the similarity of the gene expression structure between the simulated data
        vs simulated data + technical variation.  

    Arguments
    ----------
    run: int
        Unique core identifier that is used to create unique filenames for intermediate files

    NN_architecture: str
        Name of neural network architecture to use.
        Format 'NN_<intermediate layer>_<latent layer>'

    dataset_name: str
        Either "Human_analysis" or "Pseudomonas_analysis"

    analysis_name: str
        Parent directory where simulated data with experiments/partitionings will be stored.
        Format of the directory name is <dataset>_<sample/experiment>_lvl_sim 

    num_simulated_samples: int
        Number of samples to simulate

    lst_num_experiments: list
        List of different numbers of experiments to add to
        simulated data.  These are the number of sources of 
        technical variation that are added to the simulated
        data

    corrected: bool
        True if correction was applied

    use_pca: bool
        True if want to represent expression data in top PCs before
        calculating similarity

    num_PCs: int
        Number of top PCs to use to represent expression data

    file_prefix: str
        File prefix to determine whether to use data before correction ("Experiment" or "Partition")
        or after correction ("Experiment_corrected" or "Parition_corrected")

    input_file: str
        File name containing normalized gene expressiond data

    Returns
    --------
    similarity_score_df: df
        Similarity scores for each number of experiment/partition added per run

    permuted_scre: df
        Similarity score comparing the permuted data to the simulated data per run

    '''

  # Main

  # Generate simulated data
  simulated_data = generate_data_parallel.simulate_data(input_file,
                                                        NN_architecture,
                                                        dataset_name,
                                                        analysis_name,
                                                        num_simulated_samples,
                                                        local_dir,
                                                        base_dir)

  # Permute simulated data to be used as a negative control
  permuted_data = generate_data_parallel.permute_data(simulated_data)

  if not corrected:
    # Add technical variation
    generate_data_parallel.add_experiments_io(simulated_data,
                                              lst_num_experiments,
                                              run,
                                              local_dir,
                                              dataset_name,
                                              analysis_name)

  if corrected:
    # Remove technical variation
    generate_data_parallel.apply_correction_io(local_dir,
                                               run,
                                               dataset_name,
                                               analysis_name,
                                               lst_num_experiments,
                                               correction_method)

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
                                                                         dataset_name,
                                                                         analysis_name)

  # Convert similarity scores to pandas dataframe
  similarity_score_df = pd.DataFrame(data={'score': batch_scores},
                                     index=lst_num_experiments,
                                     columns=['score'])

  similarity_score_df.index.name = 'number of experiments'
  similarity_score_df

  # Return similarity scores and permuted score
  return permuted_score, similarity_score_df


def experiment_level_simulation(run,
                                NN_architecture,
                                dataset_name,
                                analysis_name,
                                num_simulated_experiments,
                                lst_num_partitions,
                                corrected,
                                correction_method,
                                use_pca,
                                num_PCs,
                                file_prefix,
                                input_file,
                                experiment_ids_file,
                                sample_id_colname,
                                local_dir,
                                base_dir):
  '''
    This function performs runs series of scripts that performs the following steps:
    1. Simulate gene expression data, keeping track of which sample is associated
        with a given experiment
    2. Add varying numbers of technical variation
    3. Compare the similarity of the gene expression structure between the simulated data
        vs simulated data + technical variation. 

    Arguments
    ----------
    run: int
        Unique core identifier that is used to create unique filenames for intermediate files

    NN_architecture: str
        Name of neural network architecture to use.
        Format 'NN_<intermediate layer>_<latent layer>'

    dataset_name: str
        Either "Human_analysis" or "Pseudomonas_analysis"

    analysis_name: str
        Parent directory where simulated data with experiments/partitionings will be stored.
        Format of the directory name is <dataset>_<sample/experiment>_lvl_sim 

    num_simulated_samples: int
        Number of samples to simulate

    lst_num_experiments: list
        List of different numbers of partitions to add to
        simulated data.  These are the number of sources of 
        technical variation that are added to the simulated
        data

    corrected: bool
        True if correction was applied

    use_pca: bool
        True if want to represent expression data in top PCs before
        calculating similarity

    num_PCs: int
        Number of top PCs to use to represent expression data

    file_prefix: str
        File prefix to determine whether to use data before correction ("Experiment" or "Partition")
        or after correction ("Experiment_corrected" or "Parition_corrected")

    input_file: str
        File name containing normalized gene expressiond data

    experiment_ids_file: str
        File containing all cleaned experiment ids

    Returns
    --------
    similarity_score_df: df
        Similarity scores for each number of experiment/partition added per run

    permuted_scre: df
        Similarity score comparing the permuted data to the simulated data per run
    '''

  # Main

  # Generate simulated data
  simulated_data = generate_data_parallel.simulate_compendium(num_simulated_experiments,
                                                              input_file,
                                                              NN_architecture,
                                                              dataset_name,
                                                              analysis_name,
                                                              experiment_ids_file,
                                                              sample_id_colname,
                                                              local_dir,
                                                              base_dir)

  # Permute simulated data to be used as a negative control
  permuted_data = generate_data_parallel.permute_data(simulated_data)

  if not corrected:
    # Add technical variation
    generate_data_parallel.add_experiments_grped_io(simulated_data,
                                                    lst_num_partitions,
                                                    run,
                                                    local_dir,
                                                    dataset_name,
                                                    analysis_name)

  if corrected:
    # Remove technical variation
    generate_data_parallel.apply_correction_io(local_dir,
                                               run,
                                               dataset_name,
                                               analysis_name,
                                               lst_num_partitions,
                                               correction_method)

 # Calculate similarity between compendium and compendium + noise
  batch_scores, permuted_score = similarity_metric_parallel.sim_svcca_io(simulated_data,
                                                                         permuted_data,
                                                                         corrected,
                                                                         file_prefix,
                                                                         run,
                                                                         lst_num_partitions,
                                                                         use_pca,
                                                                         num_PCs,
                                                                         local_dir,
                                                                         dataset_name,
                                                                         analysis_name)

  # Convert similarity scores to pandas dataframe
  similarity_score_df = pd.DataFrame(data={'score': batch_scores},
                                     index=lst_num_partitions,
                                     columns=['score'])

  similarity_score_df.index.name = 'number of partitions'
  similarity_score_df

  # Return similarity scores and permuted score
  return permuted_score, similarity_score_df
