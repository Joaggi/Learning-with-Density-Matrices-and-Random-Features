import pytest
from product_dict import product_dict


def test_create_empty():
    diccionarios = product_dict({}, {})

    assert not diccionarios 

def test_create_product_dict():
   original_dict = {
    "n_components": 1000,
    "c": 2**1,
    "tol": 1e-05
   }

   
   product_dict = { "gamma" : [2**i for i in range(-10,10)]}

   
