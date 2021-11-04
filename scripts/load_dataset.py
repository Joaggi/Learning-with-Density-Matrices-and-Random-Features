from load_usps import load_usps
from load_letters import load_letters
from load_gisette import load_gisette
from load_forest import load_forest
from load_mnist import load_mnist
from load_cifar import load_cifar


def load_dataset(dataset):
    if(dataset == "usps"):
        return load_usps("data/usps/usps.h5")

    if(dataset == "letters"):
        return load_letters("data/letters/letter-recognition.data")

    if(dataset == "gisette"):
        print("gisette")
        return load_gisette("data/gisette/")

    if(dataset == "forest"):
        return load_forest("data/forest/covtype.data.gz")

    if(dataset == "mnist"):
        return load_mnist("")

    if(dataset == "cifar"):
        return load_cifar("")
    
