"""
Author: Alexandra Lee
Date Created: 11 November 2019

Scripts to run simulation different types of simulations
(sample-level or experiment-level)
using functions in `generate_data_parallel.py`
"""
from simulate_expression_compendia_modules import (
    similarity_metric_parallel,
    generate_data_parallel,
)
from ponyo import simulate_expression_data
import pandas as pd
import numpy as np
import warnings


def fxn():
    warnings.warn("deprecated", DeprecationWarning)


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()


def sample_level_simulation(
    run,
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
    base_dir,
):
    """
    This function performs runs series of scripts that performs the following steps:
    1. Simulate gene expression data, ignorning the sample-experiment relationship
    2. Add varying numbers of technical variation
    3. Compare the similarity of the gene expression structure between the simulated data
        vs simulated data + technical variation.
    4. Simulate gene expression data, ignorning the sample-experiment relationship (different
        than data in 1)
    5. Add varying numbers of technical variation and apply noise correction
    6. Compare the similarity of the gene expression structure between the simulated data
        vs simulated data + noise correction.

    Arguments
    ----------
    run: int
        Unique core identifier that is used to create unique filenames for intermediate files

    NN_architecture: str
        Name of neural network architecture to use.
        Format 'NN_<intermediate layer>_<latent layer>'

    dataset_name: str
        Name for analysis directory. Either "Human" or "Pseudomonas"

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

    correction_method: str
        Noise correction method to use. Either 'limma' or 'combat

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

    local_dir: str
        Parent directory on local machine to store intermediate results

    base_dir: str
        Root directory containing analysis subdirectories

    Returns
    --------
    similarity_score_df: df
        Similarity scores for each number of experiment/partition added per run

    permuted_scre: df
        Similarity score comparing the permuted data to the simulated data per run

    """

    # Generate simulated data
    # Note: We are simulating the data twice - once for the uncorrected and once for
    # the corrected steps
    # In this case we would ideally like to use the same compendia
    # This code was originally structured to treat the uncorrected and corrected
    # as two separate analyses. This can be changed by following the example in the
    # `experiment_effect_simulation` function and `run_experiment_effect_simulation` in
    # pipeline.py. However we don't believe the trends
    # should change significantly having a matched compendia vs non-matched.
    simulated_data = simulate_expression_data.simulate_by_random_sampling(
        input_file,
        NN_architecture,
        dataset_name,
        analysis_name,
        num_simulated_samples,
        local_dir,
        base_dir,
    )

    # Permute simulated data to be used as a negative control
    permuted_data = generate_data_parallel.permute_data(simulated_data)

    if not corrected:
        # Add technical variation
        generate_data_parallel.add_experiments_io(
            simulated_data,
            lst_num_experiments,
            run,
            local_dir,
            dataset_name,
            analysis_name,
        )

    if corrected:
        # Remove technical variation
        generate_data_parallel.apply_correction_io(
            local_dir,
            run,
            dataset_name,
            analysis_name,
            lst_num_experiments,
            correction_method,
        )

    # Calculate similarity between compendium and compendium + noise
    batch_scores, permuted_score = similarity_metric_parallel.sim_svcca_io(
        simulated_data,
        permuted_data,
        corrected,
        file_prefix,
        run,
        lst_num_experiments,
        use_pca,
        num_PCs,
        local_dir,
        dataset_name,
        analysis_name,
    )

    # Convert similarity scores to pandas dataframe
    similarity_score_df = pd.DataFrame(
        data={"score": batch_scores}, index=lst_num_experiments, columns=["score"]
    )

    similarity_score_df.index.name = "number of experiments"
    similarity_score_df

    # Return similarity scores and permuted score
    return permuted_score, similarity_score_df


def experiment_level_simulation(
    run,
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
    base_dir,
):
    """
    This function performs runs series of scripts that performs the following steps:
    1. Simulate gene expression data, keeping track of which sample is associated
        with a given experiment
    2. Add varying numbers of technical variation
    3. Compare the similarity of the gene expression structure between the simulated data
        vs simulated data + technical variation.
    4. Simulate gene expression data, keeping track of which sample is associated
        with a given experiment (different than data in 1)
    5. Add varying numbers of technical variation and apply noise correction
    6. Compare the similarity of the gene expression structure between the simulated data
        vs simulated data + noise correction.

    Arguments
    ----------
    run: int
        Unique core identifier that is used to create unique filenames for intermediate files

    NN_architecture: str
        Name of neural network architecture to use.
        Format 'NN_<intermediate layer>_<latent layer>'

    dataset_name: str
        Name for analysis directory. Either "Human" or "Pseudomonas"

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

    correction_method: str
        Noise correction method to use. Either 'limma' or 'combat

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

    sample_id_colname: str
        Column header that contains sample id that maps expression data and metadata

    local_dir: str
        Parent directory on local machine to store intermediate results

    base_dir: str
        Root directory containing analysis subdirectories

    Returns
    --------
    similarity_score_df: df
        Similarity scores for each number of experiment/partition added per run

    permuted_scre: df
        Similarity score comparing the permuted data to the simulated data per run
    """

    # Generate simulated data
    # Note: We are simulating the data twice - once for the uncorrected and once for
    # the corrected steps
    # In this case we would ideally like to use the same compendia
    # This code was originally structured to treat the uncorrected and corrected
    # as two separate analyses. This can be changed by following the example in the
    # `experiment_effect_simulation` function and `run_experiment_effect_simulation` in
    # pipeline.py. However we don't believe the trends
    # should change significantly having a matched compendia vs non-matched.
    simulated_data = simulate_expression_data.simulate_by_latent_transformation(
        num_simulated_experiments,
        input_file,
        NN_architecture,
        dataset_name,
        analysis_name,
        experiment_ids_file,
        sample_id_colname,
        local_dir,
        base_dir,
    )

    # Permute simulated data to be used as a negative control
    permuted_data = generate_data_parallel.permute_data(simulated_data)

    if not corrected:
        # Add technical variation
        generate_data_parallel.add_experiments_grped_io(
            simulated_data,
            lst_num_partitions,
            run,
            local_dir,
            dataset_name,
            analysis_name,
        )

    if corrected:
        # Remove technical variation
        generate_data_parallel.apply_correction_io(
            local_dir,
            run,
            dataset_name,
            analysis_name,
            lst_num_partitions,
            correction_method,
        )

    # Calculate similarity between compendium and compendium + noise
    batch_scores, permuted_score = similarity_metric_parallel.sim_svcca_io(
        simulated_data,
        permuted_data,
        corrected,
        file_prefix,
        run,
        lst_num_partitions,
        use_pca,
        num_PCs,
        local_dir,
        dataset_name,
        analysis_name,
    )

    # Convert similarity scores to pandas dataframe
    similarity_score_df = pd.DataFrame(
        data={"score": batch_scores}, index=lst_num_partitions, columns=["score"]
    )

    similarity_score_df.index.name = "number of partitions"

    # Return similarity scores and permuted score
    return permuted_score, similarity_score_df


def experiment_effect_simulation(
    run,
    NN_architecture,
    dataset_name,
    analysis_name,
    num_simulated_experiments,
    lst_num_partitions,
    correction_method,
    use_pca,
    num_PCs,
    input_file,
    experiment_ids_file,
    sample_id_colname,
    local_dir,
    base_dir,
):
    """
    This function performs runs series of scripts that performs the following steps:
    1. Simulate gene expression data, with one experiment per partition
    2. Add varying numbers of technical variation
    3. Compare the similarity of the gene expression structure between the simulated data
        vs simulated data + technical variation.
    2. Using the same simulated data, add varying numbers of technical variation and apply
        noise correction
    3. Compare the similarity of the gene expression structure between the simulated data
        vs simulated data + noise corrected.
    

    Arguments
    ----------
    run: int
        Unique core identifier that is used to create unique filenames for intermediate files

    NN_architecture: str
        Name of neural network architecture to use.
        Format 'NN_<intermediate layer>_<latent layer>'

    dataset_name: str
        Name for analysis directory. Either "Human" or "Pseudomonas"

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

    correction_method: str
        Noise correction method to use. Either 'limma' or 'combat

    use_pca: bool
        True if want to represent expression data in top PCs before
        calculating similarity

    num_PCs: int
        Number of top PCs to use to represent expression data

    input_file: str
        File name containing normalized gene expressiond data

    experiment_ids_file: str
        File containing all cleaned experiment ids

    sample_id_colname: str
        Column header that contains sample id that maps expression data and metadata

    local_dir: str
        Parent directory on local machine to store intermediate results

    base_dir: str
        Root directory containing analysis subdirectories

    Returns
    --------
    similarity_score_df: df
        Similarity scores for each number of experiment/partition added per run

    permuted_scre: df
        Similarity score comparing the permuted data to the simulated data per run
    """
    # Generate simulated data
    # Note: Unlike the other simulations, we are using the same simulated dataset
    # for the uncorrected and corrected analysis.
    np.random.seed(run * 3)
    simulated_data = simulate_expression_data.simulate_by_latent_transformation(
        num_simulated_experiments,
        input_file,
        NN_architecture,
        dataset_name,
        analysis_name,
        experiment_ids_file,
        sample_id_colname,
        local_dir,
        base_dir,
    )

    # Permute simulated data to be used as a negative control
    permuted_data = generate_data_parallel.permute_data(simulated_data)

    # Add technical variation
    generate_data_parallel.add_experiments_grped_io(
        simulated_data, lst_num_partitions, run, local_dir, dataset_name, analysis_name,
    )

    file_prefix = "Partition"
    corrected = False
    # Calculate similarity between compendium and compendium + noise
    batch_scores, permuted_score = similarity_metric_parallel.sim_svcca_io(
        simulated_data,
        permuted_data,
        corrected,
        file_prefix,
        run,
        lst_num_partitions,
        use_pca,
        num_PCs,
        local_dir,
        dataset_name,
        analysis_name,
    )

    # Convert similarity scores to pandas dataframe
    uncorrected_similarity_score_df = pd.DataFrame(
        data={"score": batch_scores}, index=lst_num_partitions, columns=["score"]
    )

    uncorrected_similarity_score_df.index.name = "number of partitions"

    # Remove technical variation
    generate_data_parallel.apply_correction_io(
        local_dir,
        run,
        dataset_name,
        analysis_name,
        lst_num_partitions,
        correction_method,
    )

    # Calculate similarity between compendium and compendium + noise
    file_prefix = "Partition_corrected"
    corrected = True
    batch_scores, x_permuted_score = similarity_metric_parallel.sim_svcca_io(
        simulated_data,
        permuted_data,
        corrected,
        file_prefix,
        run,
        lst_num_partitions,
        use_pca,
        num_PCs,
        local_dir,
        dataset_name,
        analysis_name,
    )

    # Convert similarity scores to pandas dataframe
    corrected_similarity_score_df = pd.DataFrame(
        data={"score": batch_scores}, index=lst_num_partitions, columns=["score"]
    )

    corrected_similarity_score_df.index.name = "number of partitions"

    # Return similarity scores and permuted score
    return (
        permuted_score,
        uncorrected_similarity_score_df,
        corrected_similarity_score_df,
    )

