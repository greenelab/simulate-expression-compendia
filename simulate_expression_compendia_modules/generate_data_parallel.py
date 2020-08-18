"""
Author: Alexandra Lee
Date Created: 30 August 2019

These scripts are the components used to run each simulation experiment,
found in `simulations.py`.
These scripts use simulated data generated from ponyo (https://github.com/greenelab/ponyo)
and permute the simulated data, add noise to simulated data,
apply noise correction to simulated data, permute simulated data.
"""

import os
import random
import pandas as pd
import numpy as np
import warnings

from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

limma = importr("limma")
sva = importr("sva")
pandas2ri.activate()


def fxn():
    warnings.warn("deprecated", DeprecationWarning)


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()


def permute_data(simulated_data):
    """
    Permute the simulated data

    Arguments
    ----------
    simulated_data: df
        Dataframe containing simulated gene expression data

    Returns
    --------
    permuted simulated dataframe. This data will be used as a
    negative control in similarity analysis.
    """

    if "experiment_id" in list(simulated_data.columns):
        simulated_data_tmp = simulated_data.drop(columns="experiment_id", inplace=False)
    else:
        simulated_data_tmp = simulated_data.copy()

    # Shuffle values within each sample (row)
    # Each sample treated independently
    shuffled_simulated_arr = []
    num_samples = simulated_data.shape[0]

    for i in range(num_samples):
        row = list(simulated_data_tmp.values[i])
        shuffled_simulated_row = random.sample(row, len(row))
        shuffled_simulated_arr.append(shuffled_simulated_row)

    shuffled_simulated_data = pd.DataFrame(
        shuffled_simulated_arr,
        index=simulated_data_tmp.index,
        columns=simulated_data_tmp.columns,
    )

    return shuffled_simulated_data


def add_experiments_io(
    simulated_data, num_experiments, run, local_dir, dataset_name, analysis_name
):
    """
    Say we are interested in identifying genes that differentiate between
    disease vs normal states. However our dataset includes samples from
    different labs or protocols and there are variations
    in gene expression that are due to these other conditions
    that do not have to do with disease state.

    These non-relevant variations in the data are called technical variations
    that we want to model.  To model technical variation in our simulated data
    we will do the following:

    1. Partition our simulated data into <num_experiments>
    2. For each partition we will shift all genes using a vector of values
    sampled from a gaussian distribution centered around 0. This noise represents
    noise shared acoss the samples in the partition.
    3. Repeat this for each partition
    4. Append all shifted partitions together

    Arguments
    ----------
    simulated_data: df
        Dataframe containing simulated gene expression data

    num_experiments: list
        List of different numbers of experiments to add to
        simulated data

    run: int
        Unique core identifier that is used to create unique filenames for intermediate files

    local_dir: str
        Parent directory on local machine to store intermediate results

    dataset_name: str
        Name of analysis directory. Either "Human" or "Pseudomonas"

    analysis_name: str
        Parent directory where simulated data with experiments/partitionings are be stored.
        Format of the directory name is <dataset>_<sample/experiment>_lvl_sim

    Output
    --------
    Files of simulated data with different numbers of experiments added are save to file.
    Each file is named as "Experiment_<number of experiments added>"
    """
    analysis_dir = os.path.join(
        local_dir, "experiment_simulated", dataset_name + "_" + analysis_name
    )

    if not os.path.exists(analysis_dir):
        print("Creating new directory: \n {}".format(analysis_dir))
        os.makedirs(analysis_dir, exist_ok=True)

    # Add batch effects
    num_genes = simulated_data.shape[1]

    # Create an array of the simulated data indices
    simulated_ind = np.array(simulated_data.index)

    for i in num_experiments:
        print("Creating simulated data with {} experiments..".format(i))

        experiment_file = os.path.join(
            analysis_dir, "Experiment_" + str(i) + "_" + str(run) + ".txt.xz"
        )

        experiment_map_file = os.path.join(
            analysis_dir, "Experiment_map_" + str(i) + "_" + str(run) + ".txt.xz"
        )

        # Create dataframe with grouping
        experiment_data_map = simulated_data.copy()

        if i == 1:
            simulated_data.to_csv(experiment_file, sep="\t", compression="xz")

            # Add experiment id to map dataframe
            experiment_data_map["experiment"] = str(i)
            experiment_data_map_df = pd.DataFrame(
                data=experiment_data_map["experiment"], index=simulated_ind.sort()
            )

            experiment_data_map_df.to_csv(
                experiment_map_file, sep="\t", compression="xz"
            )

        else:
            experiment_data = simulated_data.copy()

            # Shuffle indices
            np.random.shuffle(simulated_ind)

            # Partition indices to batch
            # Note: 'array_split' will chunk data into almost equal sized chunks.
            # Returns arrays of size N % i and one array with the remainder
            partition = np.array_split(simulated_ind, i)

            for j in range(i):
                # Scalar to shift gene expressiond data
                stretch_factor = np.random.normal(0.0, 0.2, [1, num_genes])

                # Tile stretch_factor to be able to add to batches
                num_samples_per_experiment = len(partition[j])
                stretch_factor_tile = pd.DataFrame(
                    pd.np.tile(stretch_factor, (num_samples_per_experiment, 1)),
                    index=experiment_data.loc[partition[j].tolist()].index,
                    columns=experiment_data.loc[partition[j].tolist()].columns,
                )

                # Add experiments
                experiment_data.loc[partition[j].tolist()] = (
                    experiment_data.loc[partition[j].tolist()] + stretch_factor_tile
                )

                # Add experiment id to map dataframe
                experiment_data_map.loc[partition[j], "experiment"] = str(j)

            experiment_data_map_df = pd.DataFrame(
                data=experiment_data_map["experiment"], index=simulated_ind.sort()
            )

            # Save
            experiment_data.to_csv(
                experiment_file, float_format="%.3f", sep="\t", compression="xz"
            )

            experiment_data_map_df.to_csv(
                experiment_map_file, sep="\t", compression="xz"
            )


def add_experiments_grped_io(
    simulated_data, num_partitions, run, local_dir, dataset_name, analysis_name
):
    """
    Similar to `add_experiments_io` we will model technical variation in our
    simulated data. In this case, we will keep track of which samples
    are associated with an experiment.

    To do this we will:
    1. Partition our simulated data into <num_partitions>
        Here we are keeping track of experiment id and partitioning
        such that all samples from an experiment are in the same
        partition.

        Note: Partition sizes will be different since experiment
        sizes are different per experiment.
    2. For each partition we will shift all genes using a vector of values
    sampled from a gaussian distribution centered around 0.
    3. Repeat this for each partition
    4. Append all partitions together

    This function will return the files with compendia with different numbers
    of technical variation added with one file per compendia.

    Arguments
    ----------
    simulated_data_file: str
        File containing simulated gene expression data

    num_partitions: list
        List of different numbers of partitions to add
        technical variations to

    run: int
        Unique core identifier that is used to create unique filenames for intermediate files

    local_dir: str
        Parent directory on local machine to store intermediate results

    dataset_name: str
        Name of analysis directory. Either "Human" or "Pseudomonas"

    analysis_name: str
        Parent directory where simulated data with experiments/partitionings are be stored.
        Format of the directory name is <dataset>_<sample/experiment>_lvl_sim


    Output
    --------
    Files of simulated data with different numbers of experiments added are saved to file.
    Each file is named as "Experiment_<number of experiments added>"
    """

    analysis_dir = os.path.join(
        local_dir, "partition_simulated", dataset_name + "_" + analysis_name
    )

    if not os.path.exists(analysis_dir):
        print("Creating new directory: \n {}".format(analysis_dir))
        os.makedirs(analysis_dir, exist_ok=True)

    # Add batch effects
    num_genes = simulated_data.shape[1] - 1

    # Create an array of the simulated data indices
    simulated_ind = np.array(simulated_data.index)

    for i in num_partitions:
        print("Creating simulated data with {} partitions..".format(i))

        partition_file = os.path.join(
            analysis_dir, "Partition_" + str(i) + "_" + str(run) + ".txt.xz"
        )

        partition_map_file = os.path.join(
            analysis_dir, "Partition_map_" + str(i) + "_" + str(run) + ".txt.xz"
        )

        # Create dataframe with grouping
        partition_data_map = simulated_data.copy()

        if i == 1:
            simulated_data_out = simulated_data.drop(columns="experiment_id")
            simulated_data_out.to_csv(partition_file, sep="\t", compression="xz")

            # Add experiment id to map dataframe
            partition_data_map["partition"] = str(i)

            partition_data_map_df = pd.DataFrame(
                data=partition_data_map["partition"], index=simulated_ind.sort()
            )

            partition_data_map_df.to_csv(partition_map_file, sep="\t", compression="xz")

            print(f"simulated data in UNCORRECTED with {i} partitions RUN{run}")
            print(f"UNCORRECTED with {i} partitions RUN{run}", simulated_data_out.shape)
            print(
                f"UNCORRECTED with {i} partitions RUN{run}", simulated_data_out.head()
            )

        else:
            partition_data = simulated_data.copy()

            # Shuffle experiment ids
            experiment_ids = simulated_data["experiment_id"].unique()
            np.random.shuffle(experiment_ids)

            # Partition experiment ids
            # Note: 'array_split' will chunk data into almost equal sized chunks.
            # Returns arrays of size N % i and one array with the remainder
            partition = np.array_split(experiment_ids, i)

            for j in range(i):
                # Randomly select experiment ids
                # selected_experiment_ids = partition[j]

                # Get sample ids associated with experiment ids
                sample_ids = list(
                    simulated_data[
                        simulated_data["experiment_id"].isin(partition[j])
                    ].index
                )

                # Scalar to shift gene expressiond data
                stretch_factor = np.random.normal(0.0, 0.2, [1, num_genes])

                # Tile stretch_factor to be able to add to batches
                num_samples_per_partition = len(sample_ids)

                if j == 0:
                    # Drop experiment_id label to do calculation
                    partition_data.drop(columns="experiment_id", inplace=True)

                stretch_factor_tile = pd.DataFrame(
                    pd.np.tile(stretch_factor, (num_samples_per_partition, 1)),
                    index=partition_data.loc[sample_ids].index,
                    columns=partition_data.loc[sample_ids].columns,
                )

                # Add noise to partition
                partition_data.loc[sample_ids] = (
                    partition_data.loc[sample_ids] + stretch_factor_tile
                )

                # Add partition id to map dataframe
                partition_data_map.loc[sample_ids, "partition"] = str(j)

            partition_data_map_df = pd.DataFrame(
                data=partition_data_map["partition"], index=simulated_ind.sort()
            )

            # Save
            partition_data.to_csv(
                partition_file, float_format="%.3f", sep="\t", compression="xz"
            )

            partition_data_map_df.to_csv(partition_map_file, sep="\t", compression="xz")

            print(
                f"simulated data in UNCORRECTED with {i} partitions AFTER noise added RUN {run}"
            )
            print(
                f"UNCORRECTED with {i} partitions AFTER noise added RUN {run}",
                partition,
            )
            print(
                f"UNCORRECTED with {i} partitions AFTER noise added RUN {run}",
                partition_data.shape,
            )
            print(
                f"UNCORRECTED with {i} partitions AFTER noise added RUN {run}",
                partition_data.head(),
            )


def apply_correction_io(
    local_dir, run, dataset_name, analysis_name, num_experiments, correction_method
):
    """
    This function uses the limma or sva R package to correct for the technical variation
    we added using <add_experiments_io> or <add_experiments_grped_io>

    This function will return the corrected gene expression files

    Arguments
    ----------
    local_dir: str
        Root directory where simulated data with experiments/partitionings are be stored

    run: int
        Unique core identifier that is used to create unique filenames for intermediate files

    dataset_name:
        Name of analysis directory. Either "Human" or "Pseudomonas"

    analysis_name: str
        Parent directory where simulated data with experiments/partitionings are be stored.
        Format of the directory name is <dataset_name>_<sample/experiment>_lvl_sim

    num_experiments: list
        List of different numbers of experiments/partitions to add
        technical variations to

    correction_method: str
        Noise correction method. Either "limma" or "combat"


    Returns
    --------
    Files of simulated data with different numbers of experiments added and corrected are saved to file.
    Each file is named as "Experiment_<number of experiments added>".
    Note: After the data is corrected, the dimensions are now gene x sample
    """

    for i in range(len(num_experiments)):

        if "sample" in analysis_name:
            print("Correcting for {} experiments..".format(num_experiments[i]))

            experiment_file = os.path.join(
                local_dir,
                "experiment_simulated",
                dataset_name + "_" + analysis_name,
                f"Experiment_{num_experiments[i]}_{run}.txt.xz",
            )

            experiment_map_file = os.path.join(
                local_dir,
                "experiment_simulated",
                dataset_name + "_" + analysis_name,
                f"Experiment_map_{num_experiments[i]}_{run}.txt.xz",
            )

            # Read in data
            # data transposed to form gene x sample for R package
            experiment_data = pd.read_csv(
                experiment_file, header=0, index_col=0, sep="\t"
            ).T

            experiment_map = pd.read_csv(
                experiment_map_file, header=0, index_col=0, sep="\t"
            )["experiment"]
        else:
            print("Correcting for {} Partition..".format(num_experiments[i]))

            experiment_file = os.path.join(
                local_dir,
                "partition_simulated",
                dataset_name + "_" + analysis_name,
                f"Partition_{num_experiments[i]}_{run}.txt.xz",
            )

            experiment_map_file = os.path.join(
                local_dir,
                "partition_simulated",
                dataset_name + "_" + analysis_name,
                f"Partition_map_{num_experiments[i]}_{run}.txt.xz",
            )

            # Read in data
            # data transposed to form gene x sample for R package
            experiment_data = pd.read_csv(
                experiment_file, header=0, index_col=0, sep="\t"
            ).T
            print(
                f"simulated data in CORRECTION with {num_experiments[i]} partitions BEFORE RUN {run}"
            )
            print(
                f"CORRECTION with {num_experiments[i]} partitions BEFORE RUN {run}",
                experiment_data.T.shape,
            )
            print(
                f"CORRECTION with {num_experiments[i]} partitions BEFORE RUN {run}",
                experiment_data.T.head(),
            )

            experiment_map = pd.read_csv(
                experiment_map_file, header=0, index_col=0, sep="\t"
            )["partition"]
            print(
                f"CORRECTION with {num_experiments[i]} partitions BEFORE RUN {run}",
                experiment_map,
            )

        if i == 0:
            corrected_experiment_data_df = experiment_data.copy()

        else:
            # Correct for technical variation
            if correction_method == "limma":
                corrected_experiment_data = limma.removeBatchEffect(
                    experiment_data, batch=experiment_map
                )

                # Convert R object to pandas df
                # corrected_experiment_data_df = pandas2ri.ri2py_dataframe(
                #    corrected_experiment_data)
                corrected_experiment_data_df = pd.DataFrame(corrected_experiment_data)

            if correction_method == "combat":
                corrected_experiment_data = sva.ComBat(
                    np.array(experiment_data), batch=experiment_map
                )

                # Convert R object to pandas df
                # corrected_experiment_data_df = pandas2ri.ri2py_dataframe(
                #    corrected_experiment_data
                # )
                corrected_experiment_data_df = pd.DataFrame(corrected_experiment_data)

                corrected_experiment_data_df.columns = experiment_data.columns

        if "sample" in analysis_name:
            # Write out corrected files
            experiment_corrected_file = os.path.join(
                local_dir,
                "experiment_simulated",
                dataset_name + "_" + analysis_name,
                f"Experiment_corrected_{num_experiments[i]}_{run}.txt.xz",
            )

            corrected_experiment_data_df.to_csv(
                experiment_corrected_file,
                float_format="%.3f",
                sep="\t",
                compression="xz",
            )

        else:
            # Write out corrected files
            experiment_corrected_file = os.path.join(
                local_dir,
                "partition_simulated",
                dataset_name + "_" + analysis_name,
                f"Partition_corrected_{num_experiments[i]}_{run}.txt.xz",
            )

            corrected_experiment_data_df.to_csv(
                experiment_corrected_file,
                float_format="%.3f",
                sep="\t",
                compression="xz",
            )
