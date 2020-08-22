"""
Author: Alexandra Lee
Date Created: 11 March 2020

Scripts called by analysis notebooks to run entire the entire analysis pipeline:
1. Process data
2. Run simulation experiment, described in `simulations.py`
"""

from simulate_expression_compendia_modules import simulations
from ponyo import utils
import os
import pandas as pd
import numpy as np
import math

from joblib import Parallel, delayed

# import multiprocessing

import warnings


def fxn():
    warnings.warn("deprecated", DeprecationWarning)


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()


def transpose_data(data_file, out_file):
    """
    Transpose and save expression data so that it is of the form sample x gene

    Arguments
    ----------
    data_file: str
        File containing gene expression

    out_file: str
        File containing transposed gene expression
    """
    # Read data
    data = pd.read_csv(data_file, header=0, sep="\t", index_col=0)

    data.T.to_csv(out_file, sep="\t", compression="xz")


def run_simulation(config_file, input_data_file, corrected, experiment_ids_file=None):
    """
    Runs simulation experiment: either sample-level or experiment-level; with or without
    correction. The simulation experiment that is run depends on the parameters passed
    and specified in the config file

    Arguments
    ----------
    config_file: str
        File containing user defined parameters

    input_data_file: str
        File path corresponding to input dataset to use

    corrected: bool
        True if simulation is applying noise correction

    experiment_ids_file: str
        File containing experiment ids with expression data associated generated from ```create_experiment_id_file```

    """

    # Read in config variables
    params = utils.read_config(config_file)

    # Load parameters
    dataset_name = params["dataset_name"]
    simulation_type = params["simulation_type"]
    NN_architecture = params["NN_architecture"]
    use_pca = params["use_pca"]
    num_PCs = params["num_PCs"]
    local_dir = params["local_dir"]
    correction_method = params["correction_method"]
    sample_id_colname = params["metadata_colname"]
    iterations = params["iterations"]
    num_cores = params["num_cores"]

    if "sample" in simulation_type:
        num_simulated_samples = params["num_simulated_samples"]
        lst_num_experiments = params["lst_num_experiments"]
    else:
        num_simulated_experiments = params["num_simulated_experiments"]
        lst_num_partitions = params["lst_num_partitions"]

    # Output files
    # base_dir = os.path.abspath(os.path.join(os.getcwd(), "../"))
    base_dir = os.path.abspath(os.pardir)
    if corrected:
        similarity_uncorrected_file = os.path.join(
            base_dir,
            dataset_name,
            "results",
            "saved_variables",
            f"{dataset_name}_{simulation_type}_svcca_corrected_{correction_method}.pickle",
        )

        ci_file = os.path.join(
            base_dir,
            dataset_name,
            "results",
            "saved_variables",
            f"{dataset_name}_{simulation_type}_ci_corrected_{correction_method}.pickle",
        )

    else:
        similarity_uncorrected_file = os.path.join(
            base_dir,
            dataset_name,
            "results",
            "saved_variables",
            f"{dataset_name}_{simulation_type}_svcca_uncorrected_{correction_method}.pickle",
        )

        ci_file = os.path.join(
            base_dir,
            dataset_name,
            "results",
            "saved_variables",
            f"{dataset_name}_{simulation_type}_ci_uncorrected_{correction_method}.pickle",
        )

    similarity_permuted_file = os.path.join(
        base_dir,
        dataset_name,
        "results",
        "saved_variables",
        dataset_name + "_" + simulation_type + "_permuted",
    )

    # Run multiple simulations
    if "sample" in simulation_type:
        if corrected:
            file_prefix = "Experiment_corrected"
        else:
            file_prefix = "Experiment"
        results = Parallel(n_jobs=num_cores, verbose=100)(
            delayed(simulations.sample_level_simulation)(
                i,
                NN_architecture,
                dataset_name,
                simulation_type,
                num_simulated_samples,
                lst_num_experiments,
                corrected,
                correction_method,
                use_pca,
                num_PCs,
                file_prefix,
                input_data_file,
                local_dir,
                base_dir,
            )
            for i in iterations
        )

    else:
        if corrected:
            file_prefix = "Partition_corrected"
        else:
            file_prefix = "Partition"
        results = Parallel(n_jobs=num_cores, verbose=100)(
            delayed(simulations.experiment_level_simulation)(
                i,
                NN_architecture,
                dataset_name,
                simulation_type,
                num_simulated_experiments,
                lst_num_partitions,
                corrected,
                correction_method,
                use_pca,
                num_PCs,
                file_prefix,
                input_data_file,
                experiment_ids_file,
                sample_id_colname,
                local_dir,
                base_dir,
            )
            for i in iterations
        )

    # permuted score
    permuted_score = results[0][0]

    # Concatenate output dataframes
    all_svcca_scores = pd.DataFrame()

    for i in iterations:
        all_svcca_scores = pd.concat([all_svcca_scores, results[i][1]], axis=1)

    # Get mean svcca score for each row (number of experiments)
    mean_scores = all_svcca_scores.mean(axis=1).to_frame()
    mean_scores.columns = ["score"]
    print(mean_scores)

    # Get standard dev for each row (number of experiments)
    std_scores = (all_svcca_scores.std(axis=1) / math.sqrt(10)).to_frame()
    std_scores.columns = ["score"]
    print(std_scores)

    # Get confidence interval for each row (number of experiments)
    # z-score for 95% confidence interval
    err = std_scores * 1.96

    # Get boundaries of confidence interval
    ymax = mean_scores + err
    ymin = mean_scores - err

    ci = pd.concat([ymin, ymax], axis=1)
    ci.columns = ["ymin", "ymax"]
    print(ci)

    # Pickle dataframe of mean scores scores for first run, interval
    mean_scores.to_pickle(similarity_uncorrected_file)
    ci.to_pickle(ci_file)
    np.save(similarity_permuted_file, permuted_score)


def run_experiment_effect_simulation(
    config_file,
    input_data_file,
    num_simulated_experiments,
    lst_num_partitions,
    experiment_ids_file=None,
):
    """
    Runs experiment-level simulation keeping the size of partitions
    fixed (i.e. one experiment per partition). This script examines
    the contribution of individual experiments in signal detection.

    Arguments
    ----------
    config_file: str
        File containing user defined parameters

    input_data_file: str
        File path corresponding to input dataset to use

    num_simulated_experiments: int

    lst_num_partitions: list

    experiment_ids_file: str
        File containing experiment ids with expression data associated generated from ```create_experiment_id_file```

    """

    # Read in config variables
    params = utils.read_config(config_file)

    # Load parameters
    dataset_name = params["dataset_name"]
    simulation_type = params["simulation_type"]
    NN_architecture = params["NN_architecture"]
    use_pca = params["use_pca"]
    num_PCs = params["num_PCs"]
    local_dir = params["local_dir"]
    correction_method = params["correction_method"]
    sample_id_colname = params["metadata_colname"]
    iterations = params["iterations"]
    num_cores = params["num_cores"]

    # Output files
    base_dir = os.path.abspath(os.pardir)

    # Run multiple simulations
    results = Parallel(n_jobs=num_cores, verbose=100)(
        delayed(simulations.experiment_effect_simulation)(
            i,
            NN_architecture,
            dataset_name,
            simulation_type,
            num_simulated_experiments,
            lst_num_partitions,
            correction_method,
            use_pca,
            num_PCs,
            input_data_file,
            experiment_ids_file,
            sample_id_colname,
            local_dir,
            base_dir,
        )
        for i in iterations
    )

    # permuted score
    permuted_score = results[0][0]

    # Concatenate output dataframes
    uncorrected_svcca_scores = pd.DataFrame()
    corrected_svcca_scores = pd.DataFrame()

    for i in iterations:
        # svcca_scores = pd.concat([svcca_scores, results[i][1]], axis=1)
        uncorrected_svcca_scores = pd.concat(
            [uncorrected_svcca_scores, results[i][1]], axis=1
        )
        corrected_svcca_scores = pd.concat(
            [corrected_svcca_scores, results[i][2]], axis=1
        )

    # Get mean svcca score for each row (number of experiments)
    uncorrected_mean_scores = uncorrected_svcca_scores.mean(axis=1).to_frame()
    uncorrected_mean_scores.columns = ["score"]
    corrected_mean_scores = corrected_svcca_scores.mean(axis=1).to_frame()
    corrected_mean_scores.columns = ["score"]
    print("mean uncorrected svcca scores")
    print(uncorrected_mean_scores)
    print("mean corrected svcca scores")
    print(corrected_mean_scores)

    # Get CI for each row (number of experiments)
    ci_threshold = 0.95
    alpha = 1 - ci_threshold
    offset = int(len(iterations) * (alpha / 2))

    # Get CI for uncorrected data
    ymax = []
    ymin = []
    for size_compendia in [1, num_simulated_experiments]:
        sort_scores = sorted(uncorrected_svcca_scores.loc[size_compendia])
        ymin.append(sort_scores[offset])
        if offset == 0:
            ymax.append(sort_scores[-1])
        else:
            ymax.append(sort_scores[len(iterations) - offset])

    ci_uncorrected = pd.DataFrame(
        data={"ymin": ymin, "ymax": ymax}, index=[1, num_simulated_experiments]
    )

    print("uncorrected confidence interval")
    print(ci_uncorrected)

    # Get CI for corrected data
    ymax = []
    ymin = []
    for size_compendia in [1, num_simulated_experiments]:
        sort_scores = sorted(corrected_svcca_scores.loc[size_compendia])
        ymin.append(sort_scores[offset])
        if offset == 0:
            ymax.append(sort_scores[-1])
        else:
            ymax.append(sort_scores[len(iterations) - offset])

    ci_corrected = pd.DataFrame(
        data={"ymin": ymin, "ymax": ymax}, index=[1, num_simulated_experiments]
    )

    print("corrected_confidence interval")
    print(ci_corrected)

    return (
        uncorrected_mean_scores,
        ci_uncorrected,
        permuted_score,
        corrected_mean_scores,
        ci_corrected,
    )

