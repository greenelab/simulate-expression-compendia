def read_config(filename):
    """
    Read and parse configuration file containing stored user variables

    These variables are then passed to the analysis notebooks
    """
    f = open(filename)
    config_dict = {}
    for lines in f:
        items = lines.split('\t', 1)
        config_dict[items[0]] = eval(items[1])
    return config_dict
