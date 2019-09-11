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
              base_dir,
              analysis_name):
    """
    Script used by all similarity metrics to:

    1. Read in simulated, permuted data into data
    2. Generate directory for simulated experiment data to be stored
    3. Read in simulated data with a single experiment

    Returns
    --------
    simulated_data: dataframe
        Dataframe containing simulated gene expression data

    shuffled_simulated_data: dataframe
        Dataframe containing simulated gene expression data that has been permuted

    experiment_dir: str
        Directory path where simulated experiment data will be stored

    experiment_1: dataframe
        Dataframe containing simulated gene expression data from a single experiment

    """

    # Read in data
    simulated_data = pd.read_table(
        simulated_data_file,
        header=0,
        index_col=0,
        sep='\t')

    shuffled_simulated_data = pd.read_table(
        permuted_simulated_data_file,
        header=0,
        index_col=0,
        sep='\t')

    # Experiment directory
    experiment_dir = os.path.join(
        base_dir,
        "data",
        "experiment_simulated",
        analysis_name)

    # Get experiment 1
    experiment_1_file = os.path.join(
        experiment_dir,
        "Experiment_1.txt.xz")

    experiment_1 = pd.read_table(
        experiment_1_file,
        header=0,
        index_col=0,
        sep='\t')

    return [simulated_data,
            shuffled_simulated_data,
            experiment_dir,
            experiment_1]


def sim_svcca(simulated_data_file,
              permuted_simulated_data_file,
              num_experiments,
              use_pca,
              num_PCs,
              base_dir,
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
    simulated_data_file: str
        File containing simulated gene expression data

    permuted_simulated_data_file: str
        File containing permuted simulated gene expression data

    num_experiments: list
        List of different numbers of experiments to add to
        simulated data

    use_pca: bool
        True if want to represent expression data in top PCs before
        calculating similarity

    num_PCs: int
        Number of top PCs to use to represent expression data

    base_dir: str
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

    [simulated_data, shuffled_simulated_data, experiment_dir, experiment_1] = read_data(simulated_data_file,
                                                                                        permuted_simulated_data_file,
                                                                                        base_dir,
                                                                                        analysis_name)
    output_list = []

    for i in num_experiments:
        print('Calculating SVCCA score for 1 experiment vs {} experiments..'.format(i))

        # All experiments
        experiment_other_file = os.path.join(
            experiment_dir,
            "Experiment_" + str(i) + ".txt.xz")

        experiment_other = pd.read_table(
            experiment_other_file,
            header=0,
            index_col=0,
            sep='\t')

        if use_pca:
            # PCA projection
            pca = PCA(n_components=num_PCs)

            original_data_PCAencoded = pca.fit_transform(experiment_1)

            original_data_df = pd.DataFrame(original_data_PCAencoded,
                                            index=experiment_1.index
                                            )
            # Use trained model to encode expression data into SAME latent space
            experiment_data_PCAencoded = pca.fit_transform(experiment_other)
            experiment_data_df = pd.DataFrame(experiment_data_PCAencoded,
                                              index=experiment_other.index
                                              )
        else:
            # Use trained model to encode expression data into SAME latent space
            original_data_df = experiment_1

            # Use trained model to encode expression data into SAME latent space
            experiment_data_df = experiment_other

        # Check shape: ensure that the number of samples is the same between the two datasets
        if original_data_df.shape[0] != experiment_data_df.shape[0]:
            diff = original_data_df.shape[0] - experiment_data_df.shape[0]
            original_data_df = original_data_df.iloc[:-diff, :]

        # SVCCA
        svcca_results = cca_core.get_cca_similarity(original_data_df.T,
                                                    experiment_data_df.T,
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
                                                    shuffled_simulated_data.T,
                                                    verbose=False)

        permuted_svcca = np.mean(svcca_results["cca_coef1"])

    return output_list, permuted_svcca


def sim_hausdorff(simulated_data_file,
                  permuted_simulated_data_file,
                  num_experiments,
                  use_pca,
                  num_PCs,
                  base_dir,
                  analysis_name):
    '''
    We want to determine if adding multiple simulated experiments is able to capture the
    biological signal that is present in the original data:
    How much of the simulated data with a single experiment is captured in the simulated data with multiple experiments?

    In other words, we want to compare the representation of the single simulated experiment and multiple simulated experiments.

    Note: For the representation of the simulated data, users can choose to use:  
    1. All genes
    2. PCA representation with <num_PCs> dimensions

    We will use **Hausdorff distance* to compare these two representations.

    How does it work?
    Informally, two sets are close in the Hausdorff distance if every point of either set
    is close to some point of the other set. In other words, it is the greatest of all
    the distances from a point in one set to the closest point in the other set

    Arguments
    ----------
    simulated_data_file: str
        File containing simulated gene expression data

    permuted_simulated_data_file: str
        File containing permuted simulated gene expression data

    num_experiments: list
        List of different numbers of batches to add to
                simulated data

    use_pca: bool
        True if want to represent expression data in top PCs before
        calculating similarity

    num_PCs: int
        Number of top PCs to use to represent expression data

    base_dir: str
        Parent directory containing data files

    analysis_name: str
        Name of analysis. Format 'analysis_<int>'


    Returns
    --------
    output_list: array
        Similarity scores for each number of experiments added

    permuted_dist: float
        Similarity score comparing the permuted data to the simulated data

    '''

    seed(randomState)

    [simulated_data, shuffled_simulated_data, experiment_dir, experiment_1] = read_data(simulated_data_file,
                                                                                        permuted_simulated_data_file,
                                                                                        base_dir,
                                                                                        analysis_name)

    output_list = []

    for i in num_experiments:
        print('Calculating hausdorff distance between 1 experiment vs {} experiments..'.format(i))

        # All experiments
        experiment_other_file = os.path.join(
            experiment_dir,
            "Batch_" + str(i) + ".txt.xz")

        experiment_other = pd.read_table(
            experiment_other_file,
            header=0,
            index_col=0,
            sep='\t')

        if use_pca:
            # PCA projection
            pca = PCA(n_components=num_PCs)

            original_data_PCAencoded = pca.fit_transform(experiment_1)

            original_data_df = pd.DataFrame(original_data_PCAencoded,
                                            index=experiment_1.index
                                            )
            # Use trained model to encode expression data into SAME latent space
            experiment_data_PCAencoded = pca.fit_transform(experiment_other)
            experiment_data_df = pd.DataFrame(experiment_data_PCAencoded,
                                              index=experiment_other.index
                                              )
        else:
            # Use trained model to encode expression data into SAME latent space
            original_data_df = experiment_1

            # Use trained model to encode expression data into SAME latent space
            experiment_data_df = experiment_other

        # Calculate hausdorff distance
        dist = max(directed_hausdorff(original_data_df, experiment_data_df)[0],
                   directed_hausdorff(experiment_data_df, original_data_df)[0])

        output_list.append(dist)

    # Calculate hausdorff distance using permuted data
    if use_pca:
        simulated_data_PCAencoded = pca.fit_transform(simulated_data)
        simulated_data_PCAencoded_df = pd.DataFrame(simulated_data_PCAencoded,
                                                    index=simulated_data.index
                                                    )

        shuffled_data_PCAencoded = pca.fit_transform(shuffled_simulated_data)
        shuffled_data_PCAencoded_df = pd.DataFrame(shuffled_data_PCAencoded,
                                                   index=shuffled_simulated_data.index
                                                   )

        permuted_dist = max(directed_hausdorff(simulated_data_PCAencoded_df, shuffled_data_PCAencoded_df)[0],
                            directed_hausdorff(shuffled_data_PCAencoded_df, simulated_data_PCAencoded_df)[0])

    else:
        permuted_dist = max(directed_hausdorff(simulated_data, shuffled_simulated_data)[0],
                            directed_hausdorff(shuffled_simulated_data, simulated_data)[0])

    return output_list, permuted_dist


def sim_procrustes(simulated_data_file,
                   permuted_simulated_data_file,
                   num_experiments,
                   use_pca,
                   num_PCs,
                   base_dir,
                   analysis_name):
    '''
    We want to determine if adding multiple simulated experiments is able to capture the
    biological signal that is present in the original data:
    How much of the simulated data with a single experiment is captured in the simulated data with multiple experiments?

    In other words, we want to compare the representation of the single simulated experiment and multiple simulated experiments.

    Note: For the representation of the simulated data, users can choose to use:  
    1. All genes
    2. PCA representation with <num_PCs> dimensions

    We will use **Procrustes analysis* to compare these two representations.

    How does it work? Given data1 and data2, procrustes manipulates data1 to transform it into data2
    through a series of rotations, shifts, scaling in order to minimize the sum of squares of the
    difference between the two points:  Σ(data1−data2)2 .

    Arguments
    ----------
    simulated_data_file: str
        File containing simulated gene expression data

    permuted_simulated_data_file: str
        File containing permuted simulated gene expression data

    num_experiments: list
        List of different numbers of experiments to add to
                simulated data

    use_pca: bool
        True if want to represent expression data in top PCs before
        calculating similarity

    num_PCs: int
        Number of top PCs to use to represent expression data

    base_dir: str
        Parent directory containing data files

    analysis_name: str
        Name of analysis. Format 'analysis_<int>'


    Returns
    --------
    output_list: array
        Similarity scores for each number of experiments added

    permuted_disparity: float
        Similarity score comparing the permuted data to the simulated data

    '''

    seed(randomState)

    [simulated_data, shuffled_simulated_data, experiment_dir, experiment_1] = read_data(simulated_data_file,
                                                                                        permuted_simulated_data_file,
                                                                                        base_dir,
                                                                                        analysis_name)

    output_list = []

    for i in num_experiments:
        print('Cacluating disparity of 1 experiment vs {} experiments..'.format(i))

        # All experiments
        experiment_other_file = os.path.join(
            experiment_dir,
            "Batch_" + str(i) + ".txt.xz")

        experiment_other = pd.read_table(
            experiment_other_file,
            header=0,
            index_col=0,
            sep='\t')

        if use_pca:
            # PCA projection
            pca = PCA(n_components=num_PCs)

            original_data_PCAencoded = pca.fit_transform(experiment_1)

            original_data_df = pd.DataFrame(original_data_PCAencoded,
                                            index=experiment_1.index
                                            )
            # Use trained model to encode expression data into SAME latent space
            experiment_data_PCAencoded = pca.fit_transform(experiment_other)
            experiment_data_df = pd.DataFrame(experiment_data_PCAencoded,
                                              index=experiment_other.index
                                              )
        else:
            # Use trained model to encode expression data into SAME latent space
            original_data_df = experiment_1

            # Use trained model to encode expression data into SAME latent space
            experiment_data_df = experiment_other

        # Procrustes
        mtx1, mtx2, disparity = procrustes(
            original_data_df, experiment_data_df)

        output_list.append(disparity)

    # Procrustes using permuted data
    if use_pca:
        simulated_data_PCAencoded = pca.fit_transform(simulated_data)
        simulated_data_PCAencoded_df = pd.DataFrame(simulated_data_PCAencoded,
                                                    index=simulated_data.index
                                                    )

        shuffled_data_PCAencoded = pca.fit_transform(shuffled_simulated_data)
        shuffled_data_PCAencoded_df = pd.DataFrame(shuffled_data_PCAencoded,
                                                   index=shuffled_simulated_data.index
                                                   )

        mtx1, mtx2, permuted_disparity = procrustes(
            simulated_data_PCAencoded_df, shuffled_data_PCAencoded_df)

    else:
        mtx1, mtx2, permuted_disparity = procrustes(
            simulated_data, shuffled_simulated_data)

    return output_list, permuted_disparity


def sim_cca(simulated_data_file,
            permuted_simulated_data_file,
            num_experiments,
            use_pca,
            num_PCs,
            base_dir,
            analysis_name):
    '''
    We want to determine if adding multiple simulated experiments is able to capture the
    biological signal that is present in the original data:
    How much of the simulated data with a single experiment is captured in the simulated data with multiple experiments?

    In other words, we want to compare the representation of the single simulated experiment and multiple simulated experiments.

    Note: For the representation of the simulated data, users can choose to use:  
    1. All genes
    2. PCA representation with <num_PCs> dimensions

    We will use CCA directly, as opposed to using SVCCA which uses
    the numpy library to compute the individual matrix operations (i.e. dot product, inversions).

    How does CCA work? Let A and B be the two datasets we want to compare. CCA will find a set of
    basis vectors  (w,v)  that maximizes the correlation of the two datasets A and B projected onto
    their respective bases,  corr(wTA,vTB) . In other words, we want to find the basis vectors (“space”)
    such that the projection of the data onto their respective basis vectors is highly correlated.

    Arguments
    ----------
    simulated_data_file: str
        File containing simulated gene expression data

    permuted_simulated_data_file: str
        File containing permuted simulated gene expression data

    num_experments: list
        List of different numbers of experiments to add to
        simulated data

    use_pca: bool
        True if want to represent expression data in top PCs before
        calculating similarity

    num_PCs: int
        Number of top PCs to use to represent expression data

    base_dir: str
        Parent directory containing data files

    analysis_name: str
        Name of analysis. Format 'experiment_<int'


    Returns
    --------
    output_list: array
        Similarity scores for each number of experiments added

    permuted_corrcoef: float
        Similarity score comparing the permuted data to the simulated data
    '''

    seed(randomState)

    [simulated_data, shuffled_simulated_data, experiment_dir, experiment_1] = read_data(simulated_data_file,
                                                                                        permuted_simulated_data_file,
                                                                                        base_dir,
                                                                                        analysis_name)

    output_list = []

    for i in num_experiments:
        print('Cacluating CCA of 1 experiment vs {} experiments..'.format(i))

        # All experiments
        experiment_other_file = os.path.join(
            experiment_dir,
            "Batch_" + str(i) + ".txt.xz")

        experiment_other = pd.read_table(
            experiment_other_file,
            header=0,
            index_col=0,
            sep='\t')

        if use_pca:
            # PCA projection
            pca = PCA(n_components=num_PCs)

            original_data_PCAencoded = pca.fit_transform(experiment_1)

            original_data_df = pd.DataFrame(original_data_PCAencoded,
                                            index=experiment_1.index
                                            )
            # Use trained model to encode expression data into SAME latent space
            experiment_data_PCAencoded = pca.fit_transform(experiment_other)
            experiment_data_df = pd.DataFrame(experiment_data_PCAencoded,
                                              index=experiment_other.index
                                              )
        else:
            # Use trained model to encode expression data into SAME latent space
            original_data_df = experiment_1

            # Use trained model to encode expression data into SAME latent space
            batch_data_df = experiment_other

        # CCA
        U_c, V_c = cca.fit_transform(original_data_df, experiment_data_df)
        # TOP singular value or mean singular value???
        result = np.mean(np.corrcoef(U_c.T, V_c.T))

        output_list.append(result)

    # CCA of permuted dataset (Negative control)
    if use_pca:
        simulated_data_PCAencoded = pca.fit_transform(simulated_data)
        simulated_data_PCAencoded_df = pd.DataFrame(simulated_data_PCAencoded,
                                                    index=simulated_data.index
                                                    )

        shuffled_data_PCAencoded = pca.fit_transform(shuffled_simulated_data)
        shuffled_data_PCAencoded_df = pd.DataFrame(shuffled_data_PCAencoded,
                                                   index=shuffled_simulated_data.index
                                                   )

        U_c, V_c = cca.fit_transform(
            simulated_data_PCAencoded_df, shuffled_data_PCAencoded_df)
        permuted_corrcoef = np.mean(np.corrcoef(U_c.T, V_c.T))

    else:
        U_c, V_c = cca.fit_transform(simulated_data, shuffled_simulated_data)
        permuted_corrcoef = np.mean(np.corrcoef(U_c.T, V_c.T))

    return output_list, permuted_corrcoef
