from load_usps import usps
from load_letters import usps


def load_dataset(dataset):
    if(dataset == "usps"):
        return loadausps(dataset)

    if(dataset == "letters"):
        return load_letters(dataset)

    if(dataset == "gisette"):
        pass
