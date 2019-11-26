'''
Author: Alexandra Lee
Date Created: 30 August 2019

Scripts to generate simulated data, simulated data with different numbers of experiments, permuted version of simulated data
'''

import os
import ast
import pandas as pd
import numpy as np
import random
import glob
import pickle
from keras.models import load_model
from sklearn import preprocessing

import rpy2.robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
limma = importr('limma')
pandas2ri.activate()

import warnings
warnings.filterwarnings(action='ignore')


def get_sample_ids(experiment_id):
    '''
    Return sample ids for a given experiment id

    '''
    base_dir = os.path.abspath(os.path.join(os.getcwd(), "../.."))

    # metadata file
    mapping_file = os.path.join(
        base_dir,
        "data",
        "metadata",
        "sample_annotations.tsv")

# Read in metadata
    metadata = pd.read_table(
        mapping_file,
        header=0,
        sep='\t',
        index_col=0)

    selected_metadata = metadata.loc[experiment_id]
    sample_ids = list(selected_metadata['ml_data_source'])
    return sample_ids


def simulate_compendium(
    experiment_ids_file,
    num_simulated_experiments,
    normalized_data_file,
    NN_architecture,
    analysis_name
):
    '''
    Generate simulated data by randomly sampling some number of experiments
    and linearly shifting the gene expression in the VAE latent space.

    Workflow:
    1. Input gene expression data from 1 experiment (here we are assuming
    that there is only biological variation within this experiment)
    2. Encode this input into a latent space using the trained VAE model
    3. For each encoded feature, sample from a distribution using the
    the mean and standard deviation for that feature
    4. Decode the samples

    Arguments
    ----------
    experiment_ids_file: str
        File containing all cleaned experiment ids

    number_simulated_experiments: int
        Number of experiments to simulate

    normalized_data_file: str
        File containing normalized gene expression data

        ------------------------------| PA0001 | PA0002 |...
        05_PA14000-4-2_5-10-07_S2.CEL | 0.8533 | 0.7252 |...
        54375-4-05.CEL                | 0.7789 | 0.7678 |...
        ...                           | ...    | ...    |...

    NN_architecture: str
        Name of neural network architecture to use.
        Format 'NN_<intermediate layer>_<latent layer>'

    analysis_name: str
        Name of analysis. Format 'analysis_<int>'

    Returns
    --------
    simulated_data_file: str
        File containing simulated gene expression data

    '''

    # Create directory to output simulated data
    base_dir = os.path.abspath(os.path.join(os.getcwd(), "../.."))
    local_dir = os.path.abspath(os.path.join(os.getcwd(), "../../../.."))
    new_dir = os.path.join(local_dir, "Data", "Batch_effects", "simulated")

    analysis_dir = os.path.join(new_dir, analysis_name)

    if os.path.exists(analysis_dir):
        print('Directory already exists: \n {}'.format(analysis_dir))
    else:
        print('Creating new directory: \n {}'.format(analysis_dir))
    os.makedirs(analysis_dir, exist_ok=True)

    print('\n')

    # Files
    NN_dir = base_dir + "/models/" + NN_architecture
    latent_dim = NN_architecture.split('_')[-1]

    model_encoder_file = glob.glob(os.path.join(
        NN_dir,
        "*_encoder_model.h5"))[0]

    weights_encoder_file = glob.glob(os.path.join(
        NN_dir,
        "*_encoder_weights.h5"))[0]

    model_decoder_file = glob.glob(os.path.join(
        NN_dir,
        "*_decoder_model.h5"))[0]

    weights_decoder_file = glob.glob(os.path.join(
        NN_dir,
        "*_decoder_weights.h5"))[0]

    # Load saved models
    loaded_model = load_model(model_encoder_file)
    loaded_decode_model = load_model(model_decoder_file)

    loaded_model.load_weights(weights_encoder_file)
    loaded_decode_model.load_weights(weights_decoder_file)

    # Read data
    experiment_ids = pd.read_table(
        experiment_ids_file,
        header=0,
        sep='\t',
        index_col=0)

    normalized_data = pd.read_table(
        normalized_data_file,
        header=0,
        sep='\t',
        index_col=0).T

    print("Normalized gene expression data contains {} samples and {} genes".format(
        normalized_data.shape[0], normalized_data.shape[1]))

    # Simulate data

    simulated_data_df = pd.DataFrame()

    for i in range(num_simulated_experiments):

        selected_experiment_id = np.random.choice(
            experiment_ids['experiment_id'], size=1)[0]

        # Get corresponding sample ids
        sample_ids = get_sample_ids(selected_experiment_id)

        # Remove any missing sample ids
        sample_ids = list(filter(str.strip, sample_ids))

        # Remove any sample_ids that are not found in gene expression data
        # There are some experiments where most samples have gene expression but a few do not
        sample_ids = [
            sample for sample in sample_ids if sample in normalized_data.index]

        # Gene expression data for selected samples
        selected_data_df = normalized_data.loc[sample_ids]

        # Encode selected experiment into latent space
        data_encoded = loaded_model.predict_on_batch(selected_data_df)
        data_encoded_df = pd.DataFrame(
            data_encoded, index=selected_data_df.index)

        # Get centroid of original data
        centroid = data_encoded_df.mean(axis=0)

        # Add individual vectors(centroid, sample point) to new_centroid

        # Encode original gene expression data into latent space
        data_encoded_all = loaded_model.predict_on_batch(
            normalized_data)
        data_encoded_all_df = pd.DataFrame(
            data_encoded_all, index=normalized_data.index)

        data_encoded_all_df.head()

        # Find a new location in the latent space by sampling from the latent space
        encoded_means = data_encoded_all_df.mean(axis=0)
        encoded_stds = data_encoded_all_df.std(axis=0)

        latent_dim = int(latent_dim)
        new_centroid = np.zeros(latent_dim)

        for j in range(latent_dim):
            new_centroid[j] = np.random.normal(
                encoded_means[j], encoded_stds[j])

        shift_vec_df = new_centroid - centroid

        simulated_data_encoded_df = data_encoded_df.apply(
            lambda x: x + shift_vec_df, axis=1)

        # Decode simulated data into raw gene space
        simulated_data_decoded = loaded_decode_model.predict_on_batch(
            simulated_data_encoded_df)

        simulated_data_decoded_df = pd.DataFrame(simulated_data_decoded,
                                                 index=simulated_data_encoded_df.index,
                                                 columns=selected_data_df.columns)

        # Add experiment label
        simulated_data_decoded_df["experiment_id"] = selected_experiment_id + \
            "_" + str(i)

        # Concatenate dataframe per experiment together
        simulated_data_df = pd.concat(
            [simulated_data_df, simulated_data_decoded_df])

    # re-normalize per gene 0-1
    simulated_data_numeric_df = simulated_data_df.drop(
        columns=['experiment_id'], inplace=False)

    simulated_data_scaled = preprocessing.MinMaxScaler(
    ).fit_transform(simulated_data_numeric_df)

    simulated_data_scaled_df = pd.DataFrame(simulated_data_scaled,
                                            columns=simulated_data_numeric_df.columns,
                                            index=simulated_data_numeric_df.index)

    simulated_data_scaled_df['experiment_id'] = simulated_data_df['experiment_id']

    # If sampling with replacement, then there will be multiple sample ids that are the same
    # therefore we want to reset the index.
    simulated_data_scaled_df.reset_index(drop=True, inplace=True)

    print(simulated_data_scaled_df.shape)

    # Remove expression data for samples that have duplicate sample id across
    # different experiment ids
    # We remove these because we are not sure which experiment the sample should
    # belong to
    # simulated_data_scaled_df = simulated_data_scaled_df.loc[~simulated_data_scaled_df.index.duplicated(
    #    keep=False)]

    print("Return: simulated gene expression data containing {} samples and {} genes".format(
        simulated_data_scaled_df.shape[0], simulated_data_scaled_df.shape[1]))

    # Save
    simulated_data_file = os.path.join(
        local_dir,
        "Data",
        "Batch_effects",
        "simulated",
        analysis_name,
        "simulated_data.txt.xz")

    simulated_data_scaled_df.to_csv(
        simulated_data_file, float_format='%.3f', sep='\t', compression='xz')


def simulate_data(
        normalized_data_file,
        NN_architecture,
        analysis_name,
        num_simulated_samples
):
    '''
    Generate simulated data by sampling from VAE latent space.

    Workflow:
    1. Input gene expression data from 1 experiment (here we are assuming
    that there is only biological variation within this experiment)
    2. Encode this input into a latent space using the trained VAE model
    3. For each encoded feature, sample from a distribution using the
    the mean and standard deviation for that feature
    4. Decode the samples

    Arguments
    ----------
    normalized_data_file: str
        File containing normalized gene expression data

        ------------------------------| PA0001 | PA0002 |...
        05_PA14000-4-2_5-10-07_S2.CEL | 0.8533 | 0.7252 |...
        54375-4-05.CEL                | 0.7789 | 0.7678 |...
        ...                           | ...    | ...    |...

    NN_architecture: str
        Name of neural network architecture to use.
        Format 'NN_<intermediate layer>_<latent layer>'

    analysis_name: str
        Name of analysis. Format 'analysis_<int>'

    number_simulated_samples: int
        Number of samples to simulate

    Returns
    --------
    simulated_data_file: str
        File containing simulated gene expression data

    '''

    # Create directory to output simulated data
    base_dir = os.path.abspath(os.path.join(os.getcwd(), "../.."))
    local_dir = os.path.abspath(os.path.join(os.getcwd(), "../../../.."))
    new_dir = os.path.join(local_dir, "Data", "Batch_effects", "simulated")

    analysis_dir = os.path.join(new_dir, analysis_name)

    if os.path.exists(analysis_dir):
        print('Directory already exists: \n {}'.format(analysis_dir))
    else:
        print('Creating new directory: \n {}'.format(analysis_dir))
    os.makedirs(analysis_dir, exist_ok=True)

    print('\n')

    # Files
    NN_dir = base_dir + "/models/" + NN_architecture
    model_encoder_file = glob.glob(os.path.join(
        NN_dir,
        "*_encoder_model.h5"))[0]

    weights_encoder_file = glob.glob(os.path.join(
        NN_dir,
        "*_encoder_weights.h5"))[0]

    model_decoder_file = glob.glob(os.path.join(
        NN_dir,
        "*_decoder_model.h5"))[0]

    weights_decoder_file = glob.glob(os.path.join(
        NN_dir,
        "*_decoder_weights.h5"))[0]

    # Load saved models
    loaded_model = load_model(model_encoder_file)
    loaded_decode_model = load_model(model_decoder_file)

    loaded_model.load_weights(weights_encoder_file)
    loaded_decode_model.load_weights(weights_decoder_file)

    # Read data
    normalized_data = pd.read_table(
        normalized_data_file,
        header=0,
        sep='\t',
        index_col=0).T

    print("Normalized gene expression data contains {} samples and {} genes".format(
        normalized_data.shape[0], normalized_data.shape[1]))

    # Simulate data

    # Encode into latent space
    data_encoded = loaded_model.predict_on_batch(normalized_data)
    data_encoded_df = pd.DataFrame(data_encoded, index=normalized_data.index)

    latent_dim = data_encoded_df.shape[1]

    # Get mean and standard deviation per encoded feature
    encoded_means = data_encoded_df.mean(axis=0)
    encoded_stds = data_encoded_df.std(axis=0)

    # Generate samples
    new_data = np.zeros([num_simulated_samples, latent_dim])
    for j in range(latent_dim):
        # Use mean and std for feature
        new_data[:, j] = np.random.normal(
            encoded_means[j], encoded_stds[j], num_simulated_samples)

        # Use standard normal
        # new_data[:,j] = np.random.normal(0, 1, num_simulated_samples)

    new_data_df = pd.DataFrame(data=new_data)

    # Decode samples
    new_data_decoded = loaded_decode_model.predict_on_batch(new_data_df)
    simulated_data = pd.DataFrame(data=new_data_decoded)

    print("Return: simulated gene expression data containing {} samples and {} genes".format(
        simulated_data.shape[0], simulated_data.shape[1]))

    # Output
    return simulated_data
    # simulated_data_file = os.path.join(
    #    local_dir,
    #    "Data",
    #    "Batch_effects",
    #    "simulated",
    #    analysis_name,
    #    "simulated_data.txt.xz")

    # simulated_data.to_csv(
    #    simulated_data_file, float_format='%.3f', sep='\t', compression='xz')


def permute_data(simulated_data,
                 local_dir,
                 analysis_name):
    '''
    Permute the simulated data

    Arguments
    ----------
    simulated_data: df
        Dataframe containing simulated gene expression data

    local_dir: str
        Parent directory containing data files

    analysis_name: str
        Name of analysis. Format 'analysis_<int>'


    Returns
    --------
    permuted_simulated_data_file: str
        File containing permuted simulated gene expression data
        to be used as a negative control in similarity analysis.
    '''

    # Read in data
    # simulated_data = pd.read_table(
    #    simulated_data_file,
    #    header=0,
    #    index_col=0,
    #    sep='\t')

    if "experiment_id" in list(simulated_data.columns):
        simulated_data.drop(columns="experiment_id", inplace=True)

    # Shuffle values within each sample (row)
    # Each sample treated independently
    shuffled_simulated_arr = []
    num_samples = simulated_data.shape[0]

    for i in range(num_samples):
        row = list(simulated_data.values[i])
        shuffled_simulated_row = random.sample(row, len(row))
        shuffled_simulated_arr.append(shuffled_simulated_row)

    shuffled_simulated_data = pd.DataFrame(shuffled_simulated_arr,
                                           index=simulated_data.index,
                                           columns=simulated_data.columns)

    # Output
    return shuffled_simulated_data
    # permuted_simulated_data_file = os.path.join(
    #    local_dir,
    #    "Data",
    #    "Batch_effects",
    #    "simulated",
    #    analysis_name,
    #    "permuted_simulated_data.txt.xz")

    # shuffled_simulated_data.to_csv(
    #    permuted_simulated_data_file, float_format='%.3f', sep='\t', compression='xz')


def add_experiments(
        simulated_data,
        num_experiments,
        local_dir,
        analysis_name):
    '''
    Say we are interested in identifying genes that differentiate between
    disease vs normal states. However our dataset includes samples from
    different tissues or time points and there are variations
    in gene expression that are due to these other conditions
    and do not have to do with disease state.
    These non-relevant variations in the data are called batch effects.

    We want to model these batch effects. To do this we will:
    1. Partition our simulated data into n batches
    2. For each partition we will shift all genes using a vector of values
    sampled from a gaussian distribution centered around 0.
    3. Repeat this for each partition
    4. Append all batch effect partitions together

    Arguments
    ----------
    simulated_data: df
        Dataframe containing simulated gene expression data

    num_experiments: list
        List of different numbers of experiments to add to
        simulated data

    local_dir: str
        Parent directory containing data files

    analysis_name: str
        Name of analysis. Format 'analysis_<int>'


    Returns
    --------
    Files of simulated data with different numbers of experiments added.
    Each file named as "Experiment_<number of experiments added>"
    '''
    # Add batch effects
    num_simulated_samples = simulated_data.shape[0]
    num_genes = simulated_data.shape[1]

    # Create an array of the simulated data indices
    simulated_ind = np.array(simulated_data.index)

    # Create list to store multiple compendia
    ls_compendia = []

    # Create list to store mapping associated with each compendia
    ls_compendia_labels = []

    for i in num_experiments:
        print('Creating simulated data with {} experiments..'.format(i))

        # Create dataframe with grouping
        experiment_data_map = simulated_data.copy()

        if i == 1:
            ls_compendia.append(simulated_data)
            #simulated_data.to_csv(experiment_file, sep='\t', compression='xz')

            # Add experiment id to map dataframe
            experiment_data_map['experiment'] = str(i)
            experiment_data_map_df = pd.DataFrame(
                data=experiment_data_map['experiment'], index=simulated_ind.sort())

            ls_compendia_labels.append(experiment_data_map_df)
            # experiment_data_map_df.to_csv(
            #    experiment_map_file, sep='\t', compression='xz')

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
                    pd.np.tile(
                        stretch_factor,
                        (num_samples_per_experiment, 1)),
                    index=experiment_data.loc[partition[j].tolist()].index,
                    columns=experiment_data.loc[partition[j].tolist()].columns)

                # Add experiments
                experiment_data.loc[partition[j].tolist(
                )] = experiment_data.loc[partition[j].tolist()] + stretch_factor_tile

                # Add experiment id to map dataframe
                experiment_data_map.loc[partition[j], 'experiment'] = str(j)

            experiment_data_map_df = pd.DataFrame(
                data=experiment_data_map['experiment'], index=simulated_ind.sort())

            # Save
            ls_compendia.append(experiment_data)
            ls_compendia_labels.append(experiment_data_map_df)

    return ls_compendia, ls_compendia_labels


def add_experiments_io(
        simulated_data,
        num_experiments,
        run,
        local_dir,
        analysis_name):
    '''
    Say we are interested in identifying genes that differentiate between
    disease vs normal states. However our dataset includes samples from
    different tissues or time points and there are variations
    in gene expression that are due to these other conditions
    and do not have to do with disease state.
    These non-relevant variations in the data are called batch effects.

    We want to model these batch effects. To do this we will:
    1. Partition our simulated data into n batches
    2. For each partition we will shift all genes using a vector of values
    sampled from a gaussian distribution centered around 0.
    3. Repeat this for each partition
    4. Append all batch effect partitions together

    Arguments
    ----------
    simulated_data: df
        Dataframe containing simulated gene expression data

    num_experiments: list
        List of different numbers of experiments to add to
        simulated data

    local_dir: str
        Parent directory containing data files

    analysis_name: str
        Name of analysis. Format 'analysis_<int>'


    Returns
    --------
    Files of simulated data with different numbers of experiments added.
    Each file named as "Experiment_<number of experiments added>"
    '''

    # Add batch effects
    num_simulated_samples = simulated_data.shape[0]
    num_genes = simulated_data.shape[1]

    # Create an array of the simulated data indices
    simulated_ind = np.array(simulated_data.index)

    # Create list to store multiple compendia
    #ls_compendia = []

    # Create list to store mapping associated with each compendia
    #ls_compendia_labels = []

    for i in num_experiments:
        print('Creating simulated data with {} experiments..'.format(i))

        experiment_file = os.path.join(
            local_dir,
            "Data",
            "Batch_effects",
            "experiment_simulated",
            analysis_name,
            "Experiment_" + str(i) + "_" + str(run) + ".txt.xz")

        experiment_map_file = os.path.join(
            local_dir,
            "Data",
            "Batch_effects",
            "experiment_simulated",
            analysis_name,
            "Experiment_map_" + str(i) + "_" + str(run) + ".txt.xz")

        # Create dataframe with grouping
        experiment_data_map = simulated_data.copy()

        if i == 1:
            # ls_compendia.append(simulated_data)
            simulated_data.to_csv(experiment_file, sep='\t', compression='xz')

            # Add experiment id to map dataframe
            experiment_data_map['experiment'] = str(i)
            experiment_data_map_df = pd.DataFrame(
                data=experiment_data_map['experiment'], index=simulated_ind.sort())

            # ls_compendia_labels.append(experiment_data_map_df)
            experiment_data_map_df.to_csv(
                experiment_map_file, sep='\t', compression='xz')

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
                    pd.np.tile(
                        stretch_factor,
                        (num_samples_per_experiment, 1)),
                    index=experiment_data.loc[partition[j].tolist()].index,
                    columns=experiment_data.loc[partition[j].tolist()].columns)

                # Add experiments
                experiment_data.loc[partition[j].tolist(
                )] = experiment_data.loc[partition[j].tolist()] + stretch_factor_tile

                # Add experiment id to map dataframe
                experiment_data_map.loc[partition[j], 'experiment'] = str(j)

            experiment_data_map_df = pd.DataFrame(
                data=experiment_data_map['experiment'], index=simulated_ind.sort())

            # Save
            # ls_compendia.append(experiment_data)
            # ls_compendia_labels.append(experiment_data_map_df)
            experiment_data.to_csv(
                experiment_file, float_format='%.3f', sep='\t', compression='xz')

            experiment_data_map_df.to_csv(
                experiment_map_file, sep='\t', compression='xz')


def add_experiments_grped(
        simulated_data_file,
        num_partitions,
        local_dir,
        analysis_name):
    '''
    Say we are interested in identifying genes that differentiate between
    disease vs normal states. However our dataset includes samples from
    different tissues or time points and there are variations
    in gene expression that are due to these other conditions
    and do not have to do with disease state.
    These non-relevant variations in the data are called batch effects.

    We want to model these batch effects. To do this we will:
    1. Partition our simulated data into n batches
        Here we are keeping track of experiment id and partitioning
        such that all samples from an experiment are in the same
        partition.

        Note: Partition sizes will be different since experiment
        sizes are different per experiment.
    2. For each partition we will shift all genes using a vector of values
    sampled from a gaussian distribution centered around 0.
    3. Repeat this for each partition
    4. Append all batch effect partitions together

    Arguments
    ----------
    simulated_data_file: str
        File containing simulated gene expression data

    num_partitions: list
        List of different numbers of partitions to add
        technical variations to

    local_dir: str
        Parent directory containing data files

    analysis_name: str
        Name of analysis. Format 'analysis_<int>'


    Returns
    --------
    Files of simulated data with different numbers of experiments added.
    Each file named as "Experiment_<number of experiments added>"
    '''

    # Create directories
    new_dir = os.path.join(
        local_dir,
        "Data",
        "Batch_effects",
        "partition_simulated")

    analysis_dir = os.path.join(new_dir, analysis_name)

    if os.path.exists(analysis_dir):
        print('Directory already exists: \n {}'.format(analysis_dir))
    else:
        print('Creating new directory: \n {}'.format(analysis_dir))
    os.makedirs(analysis_dir, exist_ok=True)

    print('\n')

    # Read in data
    simulated_data = pd.read_table(
        simulated_data_file,
        header=0,
        index_col=0,
        compression='xz',
        sep='\t')

    # Add batch effects
    num_genes = simulated_data.shape[1] - 1

    # Create an array of the simulated data indices
    simulated_ind = np.array(simulated_data.index)

    for i in num_partitions:
        print('Creating simulated data with {} partitions..'.format(i))

        partition_file = os.path.join(
            local_dir,
            "Data",
            "Batch_effects",
            "partition_simulated",
            analysis_name,
            "Partition_" + str(i) + ".txt.xz")

        partition_map_file = os.path.join(
            local_dir,
            "Data",
            "Batch_effects",
            "partition_simulated",
            analysis_name,
            "Partition_map_" + str(i) + ".txt.xz")

        # Create dataframe with grouping
        partition_data_map = simulated_data.copy()

        if i == 1:
            simulated_data_out = simulated_data.drop(columns="experiment_id")
            simulated_data_out.to_csv(
                partition_file, sep='\t', compression='xz')

            # Add experiment id to map dataframe
            partition_data_map['partition'] = str(i)

            partition_data_map_df = pd.DataFrame(
                data=partition_data_map['partition'], index=simulated_ind.sort())

            partition_data_map_df.to_csv(
                partition_map_file, sep='\t', compression='xz')

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
                selected_experiment_ids = partition[j]

                # Get sample ids associated with experiment ids
                sample_ids = list(simulated_data[simulated_data["experiment_id"].isin(
                    partition[j])].index)

                # Scalar to shift gene expressiond data
                stretch_factor = np.random.normal(0.0, 0.2, [1, num_genes])

                # Tile stretch_factor to be able to add to batches
                num_samples_per_partition = len(sample_ids)

                if j == 0:
                    # Drop experiment_id label to do calculation
                    partition_data.drop(columns="experiment_id", inplace=True)

                stretch_factor_tile = pd.DataFrame(
                    pd.np.tile(
                        stretch_factor,
                        (num_samples_per_partition, 1)),
                    index=partition_data.loc[sample_ids].index,
                    columns=partition_data.loc[sample_ids].columns)

                # Add noise to partition
                partition_data.loc[sample_ids] = partition_data.loc[sample_ids] + \
                    stretch_factor_tile

                # Add partition id to map dataframe
                partition_data_map.loc[sample_ids, 'partition'] = str(j)

            partition_data_map_df = pd.DataFrame(
                data=partition_data_map['partition'], index=simulated_ind.sort())

            # Save
            partition_data.to_csv(
                partition_file, float_format='%.3f', sep='\t', compression='xz')

            partition_data_map_df.to_csv(
                partition_map_file, sep='\t', compression='xz')


def apply_correction(local_dir,
                     run,
                     analysis_name,
                     num_experiments,
                     ls_compendia,
                     ls_compendia_labels):

    ls_corrected_compendia = []

    for i in range(len(num_experiments)):
        print('Correcting for {} experiments..'.format(num_experiments[i]))

        # Read in data
        # data transposed to form gene x sample for R package
        experiment_data = ls_compendia[i].T

        # Read in map
        experiment_map = ls_compendia_labels[i]['experiment']

        if i == 0:
            ls_corrected_compendia.append(experiment_data)

        else:
            # Correct for technical variation
            corrected_experiment_data = limma.removeBatchEffect(
                experiment_data, batch=experiment_map)

            # Convert R object to pandas df
            corrected_experiment_data_df = pandas2ri.ri2py_dataframe(
                corrected_experiment_data)

            ls_corrected_compendia.append(corrected_experiment_data_df)

        return ls_corrected_compendia


def apply_correction_io(local_dir,
                        run,
                        analysis_name,
                        num_experiments):

    for i in range(len(num_experiments)):
        print('Correcting for {} experiments..'.format(num_experiments[i]))

        experiment_file = os.path.join(
            local_dir,
            "Data",
            "Batch_effects",
            "experiment_simulated",
            analysis_name,
            "Experiment_" + str(num_experiments[i]) + "_" + str(run) + ".txt.xz")

        experiment_map_file = os.path.join(
            local_dir,
            "Data",
            "Batch_effects",
            "experiment_simulated",
            analysis_name,
            "Experiment_map_" + str(num_experiments[i]) + "_" + str(run) + ".txt.xz")

        # Read in data
        # data transposed to form gene x sample for R package
        experiment_data = pd.read_table(
            experiment_file,
            header=0,
            index_col=0,
            sep='\t').T

        experiment_map = pd.read_table(
            experiment_map_file,
            header=0,
            index_col=0,
            sep='\t')["experiment"]

        if i == 0:
            corrected_experiment_data_df = experiment_data.copy()

        else:
            # Correct for technical variation
            corrected_experiment_data = limma.removeBatchEffect(
                experiment_data, batch=experiment_map)

            # Convert R object to pandas df
            corrected_experiment_data_df = pandas2ri.ri2py_dataframe(
                corrected_experiment_data)

        # Write out corrected files
        experiment_corrected_file = os.path.join(
            local_dir,
            "Data",
            "Batch_effects",
            "experiment_simulated",
            analysis_name,
            "Experiment_corrected_" + str(num_experiments[i]) + "_" + str(run) + ".txt.xz")

        corrected_experiment_data_df.to_csv(
            experiment_corrected_file, float_format='%.3f', sep='\t', compression='xz')
