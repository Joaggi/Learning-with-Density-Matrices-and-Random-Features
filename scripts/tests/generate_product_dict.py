import pytest
from generate_product_dict import generate_product_dict


def test_create_empty():
    diccionarios = generate_product_dict({}, {})

    assert not diccionarios 

def test_create_generate_product_dict():
   original_dict = {
    "n_components": 1000,
    "c": 2,
   }

   
   product_dict = {"gamma" : [2, 4], "iter": [10,20]}

   
   final_dict = [{'n_components': 1000, 'c': 2, 'gamma': 2, 'iter': 10}, \
    {'n_components': 1000, 'c': 2, 'gamma': 2, 'iter': 20}, \
    {'n_components': 1000, 'c': 2, 'gamma': 4, 'iter': 10}, \
    {'n_components': 1000, 'c': 2, 'gamma': 4, 'iter': 20}]


   assert final_dict == generate_product_dict(original_dict, product_dict)
