# By Alexandra Lee
# (updated October 2018)
#
# Encode gene expression data into low dimensional latent space using
# Tybalt with 2-hidden layers

import os
import argparse
import pandas as pd
import tensorflow as tf

# To ensure reproducibility using Keras during development
# https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
import numpy as np
import random as rn

from keras.layers import Input, Dense, Lambda, Layer, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
from keras import metrics, optimizers
from keras.callbacks import Callback

from functions.helper_ae import sampling_maker, CustomVariationalLayer, WarmUpCallback


def tybalt_2layer_model(
        learning_rate,
        batch_size,
        epochs,
        kappa,
        intermediate_dim,
        latent_dim,
        epsilon_std,
        rnaseq,
        base_dir,
        analysis_name):
    """
    Train 2-layer Tybalt model using input dataset

    Arguments:
    -- learning_rate: step size used for gradient descent. In other words, its how quickly the  methods is learning
    -- batch_size: Training is performed in batches. So this determines the number of samples to consider at a given time.
    -- epochs: The number of times to train over the entire input dataset.
    -- kappa: How fast to linearly ramp up KL loss 
    -- intermediate_dim: Size of the hidden layer
    -- latent_dim: Size of the bottleneck layer
    -- epsilon_std: standard deviation of Normal distribution to sample latent space
    -- rnaseq: dataframe of gene expression data
    -- base_dir: parent directory where data/, scripts/, models/ are subdirectories
    -- analysis_name: string that will be used to create a subdirectory where results and models will be stored

    Output:
        Encoding and decoding neural networks to use in downstream analysis
    """

    # The below is necessary in Python 3.2.3 onwards to
    # have reproducible behavior for certain hash-based operations.
    # See these references for further details:
    # https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
    # https://github.com/keras-team/keras/issues/2280#issuecomment-306959926
    randomState = 123
    import os
    os.environ['PYTHONHASHSEED'] = '0'

    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.

    np.random.seed(42)

    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.

    rn.seed(12345)

    # Force TensorFlow to use single thread.
    # Multiple threads are a potential source of
    # non-reproducible results.
    # For further details, see: https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res

    session_conf = tf.ConfigProto(
        intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

    from keras import backend as K

    # The below tf.set_random_seed() will make random number generation
    # in the TensorFlow backend have a well-defined initial state.
    # For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed

    tf.set_random_seed(1234)

    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)

    # Load rnaseq data
    rnaseq = rnaseq

    # Initialize hyper parameters

    original_dim = rnaseq.shape[1]
    beta = K.variable(0)

    stat_file = os.path.join(
        base_dir,
        "output",
        "stats",
        analysis_name,
        "tybalt_2layer_{}latent_stats.tsv".format(latent_dim))

    hist_plot_file = os.path.join(
        base_dir,
        "output",
        "viz",
        analysis_name,
        "tybalt_2layer_{}latent_hist.png".format(latent_dim))

    encoded_file = os.path.join(
        base_dir,
        "data",
        "encoded",
        analysis_name,
        "train_input_2layer_{}latent_encoded.txt".format(latent_dim))

    model_encoder_file = os.path.join(
        base_dir,
        "models",
        analysis_name,
        "tybalt_2layer_{}latent_encoder_model.h5".format(latent_dim))

    weights_encoder_file = os.path.join(
        base_dir,
        "models",
        analysis_name,
        "tybalt_2layer_{}latent_encoder_weights.h5".format(latent_dim))

    model_decoder_file = os.path.join(
        base_dir,
        "models",
        analysis_name,
        "tybalt_2layer_{}latent_decoder_model.h5".format(latent_dim))

    weights_decoder_file = os.path.join(
        base_dir,
        "models",
        analysis_name,
        "tybalt_2layer_{}latent_decoder_weights.h5".format(latent_dim))

    # Data initalizations

    # Split 10% test set randomly
    test_set_percent = 0.1
    rnaseq_test_df = rnaseq.sample(
        frac=test_set_percent, random_state=randomState)
    rnaseq_train_df = rnaseq.drop(rnaseq_test_df.index)

    # Create a placeholder for an encoded (original-dimensional)
    rnaseq_input = Input(shape=(original_dim, ))

    # Architecture of VAE

    # ENCODER

    # Input layer is compressed into a mean and log variance vector of size
    # `latent_dim`. Each layer is initialized with glorot uniform weights and each
    # step (dense connections, batch norm,and relu activation) are funneled
    # separately
    # Each vector of length `latent_dim` are connected to the rnaseq input tensor

    # "z_mean_dense_linear" is the encoded representation of the input
    #    Take as input arrays of shape (*, original dim) and output arrays of shape (*, latent dim)
    #    Combine input from previous layer using linear summ
    # Normalize the activations (combined weighted nodes of the previous layer)
    #   Transformation that maintains the mean activation close to 0 and the activation standard deviation close to 1.
    # Apply ReLU activation function to combine weighted nodes from previous layer
    #   relu = threshold cutoff (cutoff value will be learned)
    #   ReLU function filters noise

    # X is encoded using Q(z|X) to yield mu(X), sigma(X) that describes latent space distribution
    hidden_dense_linear = Dense(
        intermediate_dim, kernel_initializer='glorot_uniform')(rnaseq_input)
    hidden_dense_batchnorm = BatchNormalization()(hidden_dense_linear)
    hidden_encoded = Activation('relu')(hidden_dense_batchnorm)

    # Note:
    # Normalize and relu filter at each layer adds non-linear component (relu is non-linear function)
    # If architecture is layer-layer-normalization-relu then the computation is still linear
    # Add additional layers in triplicate
    z_mean_dense_linear = Dense(
        latent_dim, kernel_initializer='glorot_uniform')(hidden_encoded)
    z_mean_dense_batchnorm = BatchNormalization()(z_mean_dense_linear)
    z_mean_encoded = Activation('relu')(z_mean_dense_batchnorm)

    z_log_var_dense_linear = Dense(
        latent_dim, kernel_initializer='glorot_uniform')(rnaseq_input)
    z_log_var_dense_batchnorm = BatchNormalization()(z_log_var_dense_linear)
    z_log_var_encoded = Activation('relu')(z_log_var_dense_batchnorm)

    # Customized layer
    # Returns the encoded and randomly sampled z vector
    # Takes two keras layers as input to the custom sampling function layer with a
    # latent_dim` output
    #
    # sampling():
    # randomly sample similar points z from the latent normal distribution that is assumed to generate the data,
    # via z = z_mean + exp(z_log_sigma) * epsilon, where epsilon is a random normal tensor
    # z ~ Q(z|X)
    # Note: there is a trick to reparameterize to standard normal distribution so that the space is differentiable and
    # therefore gradient descent can be used
    #
    # Returns the encoded and randomly sampled z vector
    # Takes two keras layers as input to the custom sampling function layer with a
    # latent_dim` output
    z = Lambda(sampling_maker(epsilon_std),
               output_shape=(latent_dim, ))([z_mean_encoded, z_log_var_encoded])

    # DECODER

    # The decoding layer is much simpler with a single layer glorot uniform
    # initialized and sigmoid activation
    # Reconstruct P(X|z)
    decoder_model = Sequential()
    decoder_model.add(
        Dense(intermediate_dim, activation='relu', input_dim=latent_dim))
    decoder_model.add(Dense(original_dim, activation='sigmoid'))
    rnaseq_reconstruct = decoder_model(z)

    # CONNECTIONS
    # fully-connected network
    adam = optimizers.Adam(lr=learning_rate)
    vae_layer = CustomVariationalLayer(original_dim, z_log_var_encoded, z_mean_encoded, beta)([
        rnaseq_input, rnaseq_reconstruct])
    vae = Model(rnaseq_input, vae_layer)
    vae.compile(optimizer=adam, loss=None, loss_weights=[beta])

    # Training

    # fit Model
    # hist: record of the training loss at each epoch
    hist = vae.fit(
        np.array(rnaseq_train_df),
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(np.array(rnaseq_test_df), None),
        callbacks=[WarmUpCallback(beta, kappa)])

    # Use trained model to make predictions
    encoder = Model(rnaseq_input, z_mean_encoded)

    encoded_rnaseq_df = encoder.predict_on_batch(rnaseq)
    encoded_rnaseq_df = pd.DataFrame(encoded_rnaseq_df, index=rnaseq.index)

    encoded_rnaseq_df.columns.name = 'sample_id'
    encoded_rnaseq_df.columns = encoded_rnaseq_df.columns + 1

    # Visualize training performance
    history_df = pd.DataFrame(hist.history)
    ax = history_df.plot()
    ax.set_xlabel('Epochs')
    ax.set_ylabel('VAE Loss')
    fig = ax.get_figure()
    fig.savefig(hist_plot_file)

    del ax, fig

    # Output

    # Save training performance
    history_df = pd.DataFrame(hist.history)
    history_df = history_df.assign(learning_rate=learning_rate)
    history_df = history_df.assign(batch_size=batch_size)
    history_df = history_df.assign(epochs=epochs)
    history_df = history_df.assign(kappa=kappa)
    history_df.to_csv(stat_file, sep='\t', index=False)

    # Save latent space representation
    encoded_rnaseq_df.to_csv(encoded_file, sep='\t')

    # Save models
    # (source) https://machinelearningmastery.com/save-load-keras-deep-learning-models/
    # Save encoder model
    encoder.save(model_encoder_file)

    # serialize weights to HDF5
    encoder.save_weights(weights_encoder_file)

    # Save decoder model
    # (source) https://github.com/greenelab/tybalt/blob/master/scripts/nbconverted/tybalt_vae.py
    # can generate from any sampled z vector
    decoder_input = Input(shape=(latent_dim, ))
    _x_decoded_mean = decoder_model(decoder_input)
    decoder = Model(decoder_input, _x_decoded_mean)

    decoder.save(model_decoder_file)

    # serialize weights to HDF5
    decoder.save_weights(weights_decoder_file)

    # Save weight matrix:  how each gene contribute to each feature
    # build a generator that can sample from the learned distribution
    # can generate from any sampled z vector
    decoder_input = Input(shape=(latent_dim, ))
    x_decoded_mean = decoder_model(decoder_input)
    decoder = Model(decoder_input, x_decoded_mean)
    weights = []
    for layer in decoder.layers:
        weights.append(layer.get_weights())

    # Multiply hidden layers together to obtain a single representation of gene weights
    #intermediate_weight_df = pd.DataFrame(weights[1][0])
    #hidden_weight_df = pd.DataFrame(weights[1][2])
    #abstracted_weight_df = intermediate_weight_df.dot(hidden_weight_df)

    #abstracted_weight_df.index = range(0, latent_dim)
    #abstracted_weight_df.columns = rnaseq.columns

    # weight_file = os.path.join(
     #   base_dir,
     #   "data",
     #   analysis_name,
     #   "VAE_weight_matrix.txt")

    #abstracted_weight_df.to_csv(weight_file, sep='\t')
