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

import warnings
warnings.filterwarnings(action='ignore')

from numpy.random import seed
randomState = 123


def simulate_data(
        normalized_data_file,
        NN_architecture,
        analysis_name,
        num_simulated_samples,
        num_dims
):
    '''
    Generate simulated data by sampling from VAE latent space. Then
    truncate the simulated data to only include some number of dimensions.

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
        54375-4-05.CEL				  | 0.7789 | 0.7678 |...
        ...							  |	...	   | ...    |...

    NN_architecture: str
        Name of neural network architecture to use.  
        Format 'NN_<intermediate layer>_<latent layer>'

    analysis_name: str
        Name of analysis. Format 'analysis_<int>'

    number_simulated_samples: int
        Number of samples to simulate

    number_dims: int
        Number of feature dimensions (i.e. genes) to use
        * This will be removed in the future but wanted
        to keep parameters the same as previous runs to 
        verify scripts

    Returns
    --------
    simulated_data_file: str
        File containing simulated gene expression data

    '''
    seed(randomState)

    # Create directory to output simulated data
    base_dir = os.path.abspath(os.path.join(os.getcwd(), "../.."))
    new_dir = os.path.join(base_dir, "data", "simulated")

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

    # Randomly select subset of genes ********
    subset_simulated_data = simulated_data.sample(n=num_dims, axis=1)
    subset_simulated_data.head()

    print("Return: simulated gene expression data containing {} samples and {} genes".format(
        subset_simulated_data.shape[0], subset_simulated_data.shape[1]))

    # Output
    simulated_data_file = os.path.join(
        base_dir,
        "data",
        "simulated",
        analysis_name,
        "simulated_data.txt.xz")

    subset_simulated_data.to_csv(
        simulated_data_file, float_format='%.3f', sep='\t', compression='xz')


def permute_data(simulated_data_file,
                 base_dir,
                 analysis_name):
    '''
    Permute the simulated data

    Arguments
    ----------
    simulated_data_file: str
        File containing simulated gene expression data

    base_dir: str
        Parent directory containing data files

    analysis_name: str
        Name of analysis. Format 'analysis_<int>'


    Returns
    --------
    permuted_simulated_data_file: str
        File containing permuted simulated gene expression data
        to be used as a negative control in similarity analysis.
    '''

    seed(randomState)
    # Read in data
    simulated_data = pd.read_table(
        simulated_data_file,
        header=0,
        index_col=0,
        sep='\t')

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
    permuted_simulated_data_file = os.path.join(
        base_dir,
        "data",
        "simulated",
        analysis_name,
        "permuted_simulated_data.txt.xz")

    shuffled_simulated_data.to_csv(
        permuted_simulated_data_file, float_format='%.3f', sep='\t', compression='xz')


def add_experiments(
        simulated_data_file,
        num_experiments,
        base_dir,
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
    simulated_data_file: str
        File containing simulated gene expression data

    num_experiments: list
        List of different numbers of experiments to add to 
        simulated data

    base_dir: str
        Parent directory containing data files

    analysis_name: str
        Name of analysis. Format 'analysis_<int>'


    Returns
    --------
    Files of simulated data with different numbers of experiments added.  
    Each file named as "Experiment_<number of experiments added>"
    '''

    seed(randomState)

    # Create directories
    base_dir = os.path.abspath(os.path.join(os.getcwd(), "../.."))

    new_dir = os.path.join(
        base_dir,
        "data",
        "experiment_simulated")

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
    num_simulated_samples = simulated_data.shape[0]
    num_genes = simulated_data.shape[1]

    # Create an array of the simulated data indices
    simulated_ind = np.array(simulated_data.index)

    for i in num_experiments:
        print('Creating simulated data with {} experiments..'.format(i))

        experiment_file = os.path.join(
            base_dir,
            "data",
            "experiment_simulated",
            analysis_name,
            "Experiment_" + str(i) + ".txt.xz")

        if i == 1:
            simulated_data.to_csv(experiment_file, sep='\t', compression='xz')

        else:
            experiment_data = simulated_data.copy()

            # Shuffle indices
            np.random.shuffle(simulated_ind)

            # Partition indices to batch
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

            # Save
            experiment_data.to_csv(
                experiment_file, float_format='%.3f', sep='\t', compression='xz')
