from load_usps import load_usps
from load_letters import load_letters


def load_dataset(dataset):
    if(dataset == "usps"):
        return load_usps(dataset)

    if(dataset == "letters"):
        return load_letters(dataset)

    if(dataset == "gisette"):
        pass
