"""
Author: Alexandra Lee
Date Created: 30 August 2019

Scripts to compare simulated compendium with simulated compendia with noise added.
"""

from sklearn.decomposition import PCA
from simulate_expression_compendia_modules import cca_core
import os
import pandas as pd
import numpy as np
import warnings


def fxn():
    warnings.warn("deprecated", DeprecationWarning)


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()


def read_data(simulated_data, file_prefix, run, local_dir, dataset_name, analysis_name):
    """
    Script used by all similarity metrics to:

    1. Read in simulated data into data
    2. Generate directory where simulated experiment data is already stored
    3. Read in simulated data with a single experiment/partitioning

    Returns
    --------
    simulated_data: dataframe
        Dataframe containing simulated gene expression data

    file_prefix: str
        File prefix to determine whether to use data before correction ("Experiment" or "Partition")
        or after correction ("Experiment_corrected" or "Parition_corrected")

    run: int
        Unique core identifier that is used to create unique filenames for intermediate files

    local_dir: str
        Root directory where simulated data with experiments/partitionings are be stored

    dataset_name: str
        Name for analysis directory. Either "Human" or "Pseudomonas"

    analysis_name: str
        Parent directory where simulated data with experiments/partitionings are be stored.
        Format of the directory name is <dataset>_<sample/experiment>_lvl_sim

    """

    if "experiment_id" in list(simulated_data.columns):
        simulated_data_numeric = simulated_data.drop(columns="experiment_id")

        # Compendium directory
        compendium_dir = os.path.join(
            local_dir, "partition_simulated", dataset_name + "_" + analysis_name
        )
    else:
        simulated_data_numeric = simulated_data.copy()
        # Compendium directory
        compendium_dir = os.path.join(
            local_dir, "experiment_simulated", dataset_name + "_" + analysis_name
        )

    # Get compendium with 1 experiment or partitioning
    compendium_1_file = os.path.join(
        compendium_dir, file_prefix + "_1" + "_" + str(run) + ".txt.xz"
    )

    compendium_1 = pd.read_csv(compendium_1_file, header=0, index_col=0, sep="\t")

    # Transpose compendium df because output format
    # for correction method is swapped
    if file_prefix.split("_")[-1] == "corrected":
        compendium_1 = compendium_1.T

    return [simulated_data_numeric, compendium_dir, compendium_1]


def sim_svcca_io(
    simulated_data,
    permuted_simulated_data,
    corrected,
    file_prefix,
    run,
    num_experiments,
    use_pca,
    num_PCs,
    local_dir,
    dataset_name,
    analysis_name,
):
    """
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

    corrected: bool
        True if correction was applied

    file_prefix: str
        File prefix to determine whether to use data before correction ("Experiment" or "Partition")
        or after correction ("Experiment_corrected" or "Parition_corrected")

    run: int
        Unique core identifier that is used to create unique filenames for intermediate files

    num_experiments: list
        List of different numbers of experiments/partitions that were added to
        simulated data

    use_pca: bool
        True if want to represent expression data in top PCs before
        calculating similarity

    num_PCs: int
        Number of top PCs to use to represent expression data

    local_dir: str
        Root directory where simulated data with experiments/partitionings are be stored

    dataset_name: str
        Name for analysis directory. Either "Human" or "Pseudomonas"

    analysis_name: str
        Parent directory where simulated data with experiments/partitionings are be stored.
        Format of the directory name is <dataset>_<sample/experiment>_lvl_sim


    Returns
    --------
    output_list: array
        Similarity scores for each number of experiment/partition added

    permuted_svcca: float
        Similarity score comparing the permuted data to the simulated data

    """

    [simulated_data, compendium_dir, compendium_1] = read_data(
        simulated_data, file_prefix, run, local_dir, dataset_name, analysis_name
    )

    output_list = []

    for i in range(len(num_experiments)):
        if "sample" in analysis_name:
            print(
                "Calculating SVCCA score for 1 experiment vs {} experiments..".format(
                    num_experiments[i]
                )
            )
        else:
            print(
                "Calculating SVCCA score for 1 partition vs {} partitions..".format(
                    num_experiments[i]
                )
            )

        # All experiments/partitions
        compendium_other_file = os.path.join(
            compendium_dir,
            file_prefix + "_" + str(num_experiments[i]) + "_" + str(run) + ".txt.xz",
        )

        compendium_other = pd.read_csv(
            compendium_other_file, header=0, index_col=0, sep="\t"
        )

        # Transpose compendium df because output format
        # for correction method is swapped
        if corrected:
            compendium_other = compendium_other.T

        if use_pca:
            # PCA projection
            pca = PCA(n_components=num_PCs)

            original_data_PCAencoded = pca.fit_transform(compendium_1)

            original_data_df = pd.DataFrame(
                original_data_PCAencoded, index=compendium_1.index
            )
            # Train new PCA model to encode expression data into DIFFERENT latent space
            pca_new = PCA(n_components=num_PCs)
            noisy_original_data_PCAencoded = pca_new.fit_transform(compendium_other)
            noisy_original_data_df = pd.DataFrame(
                noisy_original_data_PCAencoded, index=compendium_other.index
            )

        else:
            original_data_df = compendium_1
            noisy_original_data_df = compendium_other

        # SVCCA
        svcca_results = cca_core.get_cca_similarity(
            original_data_df.T, noisy_original_data_df.T, verbose=False
        )

        output_list.append(np.mean(svcca_results["cca_coef1"]))

    # SVCCA of permuted data
    if use_pca:
        pca = PCA(n_components=num_PCs)
        simulated_data_PCAencoded = pca.fit_transform(simulated_data)
        simulated_data_PCAencoded_df = pd.DataFrame(
            simulated_data_PCAencoded, index=simulated_data.index
        )

        pca_new = PCA(n_components=num_PCs)
        shuffled_data_PCAencoded = pca_new.fit_transform(permuted_simulated_data)
        shuffled_data_PCAencoded_df = pd.DataFrame(
            shuffled_data_PCAencoded, index=permuted_simulated_data.index
        )

        svcca_results = cca_core.get_cca_similarity(
            simulated_data_PCAencoded_df.T, shuffled_data_PCAencoded_df.T, verbose=False
        )

        permuted_svcca = np.mean(svcca_results["cca_coef1"])

    else:
        svcca_results = cca_core.get_cca_similarity(
            simulated_data.T, permuted_simulated_data.T, verbose=False
        )

        permuted_svcca = np.mean(svcca_results["cca_coef1"])

    return output_list, permuted_svcca
