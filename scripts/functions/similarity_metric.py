'''
Author: Alexandra Lee
Date Created: 30 August 2019

Scripts to generate simulated data and visualize it
'''

import os
import sys
import ast
import pandas as pd
import numpy as np
import random
import glob
import warnings
warnings.filterwarnings(action='ignore')

sys.path.append("../")
from functions import cca_core

from sklearn.decomposition import PCA
from numpy.random import seed
randomState = 123
seed(randomState)


def read_data(simulated_data_file,
              permuted_simulated_data_file,
              file_prefix,
              local_dir,
              analysis_name):
    """
    Script used by all similarity metrics to:

    1. Read in simulated, permuted data into data
    2. Generate directory for simulated experiment data to be stored
    3. Read in simulated data with a single experiment/partitioning

    Returns
    --------
    simulated_data: dataframe
        Dataframe containing simulated gene expression data

    permuted_simulated_data: dataframe
        Dataframe containing simulated gene expression data that has been permuted

    file_prefix: str
        File prefix to determine whether to use data before correction ("Experiment" or "Partition")
        or after correction ("Experiment_corrected" or "Parition_corrected")

    compendium_dir: str
        Directory path where simulated data with experiments/partitionings will be stored

    compendium_1: dataframe
        Dataframe containing simulated gene expression data from a single experiment/partitioning

    """

    # Read in data
    simulated_data = pd.read_table(
        simulated_data_file,
        header=0,
        index_col=0,
        sep='\t')

    if "experiment_id" in list(simulated_data.columns):
        simulated_data.drop(columns="experiment_id", inplace=True)

        # Compendium directory
        compendium_dir = os.path.join(
            local_dir,
            "Data",
            "Batch_effects",
            "partition_simulated",
            analysis_name)
    else:
        # Compendium directory
        compendium_dir = os.path.join(
            local_dir,
            "Data",
            "Batch_effects",
            "experiment_simulated",
            analysis_name)

    shuffled_simulated_data = pd.read_table(
        permuted_simulated_data_file,
        header=0,
        index_col=0,
        sep='\t')

    # Get compendium with 1 experiment or partitioning
    compendium_1_file = os.path.join(
        compendium_dir,
        file_prefix + "_1.txt.xz")

    compendium_1 = pd.read_table(
        compendium_1_file,
        header=0,
        index_col=0,
        sep='\t')

    # Transpose compendium df because output format
    # for correction method is swapped
    if file_prefix.split("_")[-1] == "corrected":
        compendium_1 = compendium_1.T

    return [simulated_data,
            shuffled_simulated_data,
            compendium_dir,
            compendium_1]


def sim_svcca(simulated_data_file,
              permuted_simulated_data_file,
              file_prefix,
              num_experiments,
              use_pca,
              num_PCs,
              local_dir,
              analysis_name):
    '''
    We want to determine if adding multiple simulated experiments is able to capture the
    biological signal that is present in the original data:
    How much of the simulated data with a single experiment is captured in the simulated data with multiple experiments?

    In other words, we want to compare the representation of the single simulated experiment and multiple simulated experiments.

    Note: For the representation of the simulated data, users can choose to use:  
    1. All genes
    2. PCA representation with <num_PCs> dimensions

    We will use **SVCCA** to compare these two representations.

    How does it work?
    Singular Vector Canonical Correlation Analysis
    [Raghu et al. 2017](https://arxiv.org/pdf/1706.05806.pdf) [(github)](https://github.com/google/svcca)
    to the UMAP and PCA representations of our batch 1 simulated dataset vs batch n simulated datasets.
    The output of the SVCCA analysis is the SVCCA mean similarity score. This single number can be interpreted
    as a measure of similarity between our original data vs batched dataset.

    Briefly, SVCCA uses Singular Value Decomposition (SVD) to extract the components explaining 99% of the variation.
    This is done to remove potential dimensions described by noise. Next, SVCCA performs a Canonical Correlation Analysis (CCA)
     on the SVD matrices to identify maximum correlations of linear combinations of both input matrices.
     The algorithm will identify the canonical correlations of highest magnitude across and within algorithms of the same dimensionality.

    Arguments
    ----------
    simulated_data: df
        Dataframe containing simulated gene expression data

    permuted_simulated_data: df
        Dataframe containing permuted simulated gene expression data

    lst_compendia: list
        list of dataframes containing simulated gene expression data
        with varying amount of technical variation added

    corrected: bool
        True if correction was applied

    num_experiments: list
        List of different numbers of experiments/partitions to add to
        simulated data

    use_pca: bool
        True if want to represent expression data in top PCs before
        calculating similarity

    num_PCs: int
        Number of top PCs to use to represent expression data

    local_dir: str
        Parent directory containing data files

    analysis_name: str
        Name of analysis. Format 'analysis_<int>'


    Returns
    --------
    output_list: array
        Similarity scores for each number of experiments added

    permuted_svcca: float
        Similarity score comparing the permuted data to the simulated data

    '''

    seed(randomState)

    [simulated_data, shuffled_simulated_data, compendium_dir, compendium_1] = read_data(simulated_data_file,
                                                                                        permuted_simulated_data_file,
                                                                                        file_prefix,
                                                                                        local_dir,
                                                                                        analysis_name)

    output_list = []

    #compendium_1 = lst_compendia[0][0]

    for i in num_experiments:
        if "Experiment" in file_prefix:
            print(
                'Calculating SVCCA score for 1 experiment vs {} experiments..'.format(i))
        else:
            print('Calculating SVCCA score for 1 partition vs {} partitions..'.format(i))

        # All experiments/partitions
        #compendium_other = lst_compendia[i][0]

        compendium_other_file = os.path.join(
            compendium_dir,
            file_prefix + "_" + str(i) + ".txt.xz")

        compendium_other = pd.read_table(
            compendium_other_file,
            header=0,
            index_col=0,
            sep='\t')

        #print(compendium_other.shape)

        # Transpose compendium df because output format
        # for correction method is swapped
        if "corrected" in file_prefix:
            compendium_other = compendium_other.T

        if use_pca:
            # PCA projection
            pca = PCA(n_components=num_PCs)

            original_data_PCAencoded = pca.fit_transform(compendium_1)

            original_data_df = pd.DataFrame(original_data_PCAencoded,
                                            index=compendium_1.index
                                            )
            # Use trained model to encode expression data into SAME latent space
            noisy_original_data_PCAencoded = pca.fit_transform(
                compendium_other)
            noisy_original_data_df = pd.DataFrame(noisy_original_data_PCAencoded,
                                                  index=compendium_other.index
                                                  )
        else:
            # Use trained model to encode expression data into SAME latent space
            original_data_df = compendium_1

            # Use trained model to encode expression data into SAME latent space
            noisy_original_data_df = compendium_other

        # SVCCA
        svcca_results = cca_core.get_cca_similarity(original_data_df.T,
                                                    noisy_original_data_df.T,
                                                    verbose=False)

        output_list.append(np.mean(svcca_results["cca_coef1"]))

    # SVCCA of permuted data
    if use_pca:
        simulated_data_PCAencoded = pca.fit_transform(simulated_data)
        simulated_data_PCAencoded_df = pd.DataFrame(simulated_data_PCAencoded,
                                                    index=simulated_data.index
                                                    )

        shuffled_data_PCAencoded = pca.fit_transform(shuffled_simulated_data)
        shuffled_data_PCAencoded_df = pd.DataFrame(shuffled_data_PCAencoded,
                                                   index=shuffled_simulated_data.index
                                                   )

        svcca_results = cca_core.get_cca_similarity(simulated_data_PCAencoded_df.T,
                                                    shuffled_data_PCAencoded_df.T,
                                                    verbose=False)

        permuted_svcca = np.mean(svcca_results["cca_coef1"])

    else:
        svcca_results = cca_core.get_cca_similarity(simulated_data.T,
                                                    permuted_data.T,
                                                    verbose=False)

        permuted_svcca = np.mean(svcca_results["cca_coef1"])

    return output_list, permuted_svcca
