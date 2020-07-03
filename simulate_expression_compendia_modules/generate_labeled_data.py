"""
Author: Alexandra Lee
Date Created: 30 August 2019

Scripts to generate gene expression compendia using experiment-preserving
approach with added experiment id labels for visualizations.
These scripts are used to generate visualizations that validate this
experiment-preserving simulation approach.
"""

from numpy.random import seed
import os
import pandas as pd
import numpy as np
import glob
from keras.models import load_model
from sklearn import preprocessing
import warnings


def fxn():
    warnings.warn("deprecated", DeprecationWarning)


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()


random_state = 123
seed(random_state)


def get_sample_ids(experiment_id, dataset_name):
    """
    Return sample ids for a given experiment id

    """
    base_dir = os.path.abspath(os.path.join(os.getcwd(), "../"))

    if "pseudomonas" in dataset_name.lower():
        # metadata file
        mapping_file = os.path.join(
            base_dir, dataset_name, "data", "metadata", "sample_annotations.tsv"
        )

        # Read in metadata
        metadata = pd.read_csv(mapping_file, header=0, sep="\t", index_col=0)

        selected_metadata = metadata.loc[experiment_id]
        sample_ids = list(selected_metadata["ml_data_source"])

    else:
        # metadata file
        mapping_file = os.path.join(
            base_dir, dataset_name, "data", "metadata", "recount2_metadata.tsv"
        )

        # Read in metadata
        metadata = pd.read_csv(mapping_file, header=0, sep="\t", index_col=0)

        selected_metadata = metadata.loc[experiment_id]
        sample_ids = list(selected_metadata["run"])

    return sample_ids


def simulate_compendium_labeled(
    experiment_ids_file,
    num_simulated_experiments,
    normalized_data_file,
    NN_architecture,
    dataset_name,
    local_dir,
    base_dir,
):
    """
    Generate simulated data using a list of experiment_ids found in
    experiment_ids_file as templates. The compendia will contain the
    shifted experiments using the experiment_ids as templates and following
    the same workflow as simulate_compendia in generate_data_parallel.py

    This function adds a column to label each sample with the experiment_id that
    they originated from. This will be used for plotting.

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

    dataset_name:
        Name of analysis directory. Either "Human" or "Pseudomonas"

    local_dir: str
        Parent directory on local machine to store intermediate results

    base_dir: str
        Root directory containing analysis subdirectories

    Returns
    --------
    simulated_data_file: str
        File containing simulated gene expression data

    """
    seed(random_state)

    # Files
    NN_dir = os.path.join(base_dir, dataset_name, "models", NN_architecture)
    latent_dim = NN_architecture.split("_")[-1]

    model_encoder_file = glob.glob(os.path.join(NN_dir, "*_encoder_model.h5"))[0]

    weights_encoder_file = glob.glob(os.path.join(NN_dir, "*_encoder_weights.h5"))[0]

    model_decoder_file = glob.glob(os.path.join(NN_dir, "*_decoder_model.h5"))[0]

    weights_decoder_file = glob.glob(os.path.join(NN_dir, "*_decoder_weights.h5"))[0]

    # Load saved models
    loaded_model = load_model(model_encoder_file)
    loaded_decode_model = load_model(model_decoder_file)

    loaded_model.load_weights(weights_encoder_file)
    loaded_decode_model.load_weights(weights_decoder_file)

    # Read data
    experiment_ids = pd.read_csv(experiment_ids_file, header=0, sep="\t", index_col=0)

    normalized_data = pd.read_csv(normalized_data_file, header=0, sep="\t", index_col=0)

    print(
        "Normalized gene expression data contains {} samples and {} genes".format(
            normalized_data.shape[0], normalized_data.shape[1]
        )
    )

    # Simulate data

    simulated_data_df = pd.DataFrame()

    for i in range(num_simulated_experiments):

        selected_experiment_id = np.random.choice(
            experiment_ids["experiment_id"], size=1
        )[0]

        # Get corresponding sample ids
        sample_ids = get_sample_ids(selected_experiment_id, dataset_name)

        # Remove any missing sample ids
        sample_ids = list(filter(str.strip, sample_ids))

        # Remove any sample_ids that are not found in gene expression data
        # There are some experiments where most samples have gene expression but a
        # few do not
        sample_ids = [
            sample for sample in sample_ids if sample in normalized_data.index
        ]

        # Gene expression data for selected samples
        selected_data_df = normalized_data.loc[sample_ids]

        # Encode selected experiment into latent space
        data_encoded = loaded_model.predict_on_batch(selected_data_df)
        data_encoded_df = pd.DataFrame(data_encoded, index=selected_data_df.index)

        # Get centroid of original data
        centroid = data_encoded_df.mean(axis=0)

        # Add individual vectors(centroid, sample point) to new_centroid

        # Encode original gene expression data into latent space
        data_encoded_all = loaded_model.predict_on_batch(normalized_data)
        data_encoded_all_df = pd.DataFrame(
            data_encoded_all, index=normalized_data.index
        )

        data_encoded_all_df.head()

        # Find a new location in the latent space by sampling from the latent space
        encoded_means = data_encoded_all_df.mean(axis=0)
        encoded_stds = data_encoded_all_df.std(axis=0)

        latent_dim = int(latent_dim)
        new_centroid = np.zeros(latent_dim)

        for j in range(latent_dim):
            new_centroid[j] = np.random.normal(encoded_means[j], encoded_stds[j])

        shift_vec_df = new_centroid - centroid

        simulated_data_encoded_df = data_encoded_df.apply(
            lambda x: x + shift_vec_df, axis=1
        )

        # Decode simulated data into raw gene space
        simulated_data_decoded = loaded_decode_model.predict_on_batch(
            simulated_data_encoded_df
        )

        simulated_data_decoded_df = pd.DataFrame(
            simulated_data_decoded,
            index=simulated_data_encoded_df.index,
            columns=selected_data_df.columns,
        )

        # Add experiment label
        simulated_data_decoded_df["experiment_id"] = (
            selected_experiment_id + "_" + str(i)
        )

        # Concatenate dataframe per experiment together
        simulated_data_df = pd.concat([simulated_data_df, simulated_data_decoded_df])

    # re-normalize per gene 0-1
    simulated_data_numeric_df = simulated_data_df.drop(
        columns=["experiment_id"], inplace=False
    )

    simulated_data_scaled = preprocessing.MinMaxScaler().fit_transform(
        simulated_data_numeric_df
    )

    simulated_data_scaled_df = pd.DataFrame(
        simulated_data_scaled,
        columns=simulated_data_numeric_df.columns,
        index=simulated_data_numeric_df.index,
    )

    simulated_data_scaled_df["experiment_id"] = simulated_data_df["experiment_id"]

    # If sampling with replacement, then there will be multiple sample ids that are the same
    # therefore we want to reset the index.
    simulated_data_scaled_df.reset_index(drop=True, inplace=True)

    # Remove expression data for samples that have duplicate sample id across
    # different experiment ids
    # We remove these because we are not sure which experiment the sample should
    # belong to
    # simulated_data_scaled_df = simulated_data_scaled_df.loc[~simulated_data_scaled_df.index.duplicated(
    #    keep=False)]

    print(
        "Return: simulated gene expression data containing {} samples and {} genes".format(
            simulated_data_scaled_df.shape[0], simulated_data_scaled_df.shape[1]
        )
    )

    # Save
    simulated_data_file = os.path.join(
        local_dir, "pseudo_experiment", "simulated_data_labeled.txt.xz"
    )

    simulated_data_scaled_df.to_csv(
        simulated_data_file, float_format="%.3f", sep="\t", compression="xz"
    )
