from load_usps import load_usps
from load_letters import load_letters


def load_dataset(dataset):
    if(dataset == "usps"):
        return load_usps("data/usps/usps.h5")

    if(dataset == "letters"):
        return load_letters("data/letters/letters-recognition.data")

    if(dataset == "gisette"):
        pass
