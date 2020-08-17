def read_data_test(
    simulated_data, file_prefix, run, local_dir, dataset_name, analysis_name
):
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
        simulated_data.drop(columns="experiment_id", inplace=True)

        # Compendium directory
        compendium_dir = os.path.join(
            local_dir, "partition_simulated", dataset_name + "_" + analysis_name
        )
    else:
        # Compendium directory
        compendium_dir = os.path.join(
            local_dir, "experiment_simulated", dataset_name + "_" + analysis_name
        )

    # Get original compendium
    compendium_1_file = "data/input/train_set_normalized_processed.txt.xz"

    compendium_1 = pd.read_csv(compendium_1_file, header=0, index_col=0, sep="\t")

    return [simulated_data, compendium_dir, compendium_1]


def sim_svcca_io_test(
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

    [simulated_data, compendium_dir, compendium_1] = read_data_test(
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
            print("original compendium")
            print(original_data_df.shape)
            # Train new PCA model to encode expression data into DIFFERENT latent space
            pca_new = PCA(n_components=num_PCs)
            noisy_original_data_PCAencoded = pca_new.fit_transform(compendium_other)
            noisy_original_data_df = pd.DataFrame(
                noisy_original_data_PCAencoded, index=compendium_other.index
            )
            print("noisy compendium")
            print(noisy_original_data_df.shape)

        else:
            original_data_df = compendium_1
            noisy_original_data_df = compendium_other

        # SVCCA
        svcca_results = cca_core.get_cca_similarity(
            original_data_df, noisy_original_data_df, verbose=False
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
            simulated_data_PCAencoded_df, shuffled_data_PCAencoded_df, verbose=False
        )

        permuted_svcca = np.mean(svcca_results["cca_coef1"])

    else:
        svcca_results = cca_core.get_cca_similarity(
            simulated_data, permuted_simulated_data, verbose=False
        )

        permuted_svcca = np.mean(svcca_results["cca_coef1"])

    return output_list, permuted_svcca

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
    1. Simulate gene expression data, keeping track of which sample is associated
        with a given experiment
    2. Add varying numbers of technical variation
    3. Compare the similarity of the gene expression structure between the simulated data
        vs simulated data + technical variation.

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
    # Only simulate the data once so that the uncorrected and corrected steps
    # are using the same compendia
    print(f"Simulate a compendia with {num_simulated_experiments} experiments")
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
    print(simulated_data.shape)
    print(simulated_data.head())

    # Permute simulated data to be used as a negative control
    permuted_data = generate_data_parallel.permute_data(simulated_data)

    print("In uncorrected part")
    print("Adding partitions", lst_num_partitions)

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

    print(
        f"Second time simulating a compendia with {num_simulated_experiments} experiments"
    )
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
    print(simulated_data.shape)
    print(simulated_data.head())

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

	
for i in iterations:
        uncorrected_svcca_scores = pd.concat(
            [uncorrected_svcca_scores, results[i][1]], axis=1
        )
        corrected_svcca_scores = pd.concat(
            [corrected_svcca_scores, results[i][2]], axis=1
        )

    # Get mean svcca score for each row (number of experiments)
    uncorrected_mean_scores = uncorrected_svcca_scores.mean(axis=1).to_frame()
    uncorrected_mean_scores.columns = ["score"]
    print("uncorrected scores")
    print(uncorrected_mean_scores)

    corrected_mean_scores = corrected_svcca_scores.mean(axis=1).to_frame()
    corrected_mean_scores.columns = ["score"]
    print("corrected scores")
    print(corrected_mean_scores)

    # Get standard dev for each row (number of experiments)
    std_scores = (uncorrected_svcca_scores.std(axis=1) / math.sqrt(10)).to_frame()
    std_scores.columns = ["score"]
    print(std_scores)

    # Get confidence interval for each row (number of experiments)
    # z-score for 95% confidence interval
    err = std_scores * 1.96

    # Get boundaries of confidence interval
    ymax = uncorrected_mean_scores + err
    ymin = uncorrected_mean_scores - err

    ci_uncorrected = pd.concat([ymin, ymax], axis=1)
    ci_uncorrected.columns = ["ymin", "ymax"]
    print(ci_uncorrected)

    # Get standard dev for each row (number of experiments)
    std_scores = (corrected_svcca_scores.std(axis=1) / math.sqrt(10)).to_frame()
    std_scores.columns = ["score"]
    print("uncorrected std")
    print(std_scores)

    # Get confidence interval for each row (number of experiments)
    # z-score for 95% confidence interval
    err = std_scores * 1.96

    # Get boundaries of confidence interval
    ymax = corrected_mean_scores + err
    ymin = corrected_mean_scores - err

    ci_corrected = pd.concat([ymin, ymax], axis=1)
    ci_corrected.columns = ["ymin", "ymax"]
    print("corrected std")
    print(ci_corrected)

    return (
        uncorrected_mean_scores,
        ci_uncorrected,
        corrected_mean_scores,
        ci_corrected,
        permuted_score,
    )
[2, 3, 5, 10, 20, 30, 50, 70, 100, 200, 300, 400, 500, 600]