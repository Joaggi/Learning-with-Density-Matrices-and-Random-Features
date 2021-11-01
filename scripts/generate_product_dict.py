import itertools

def generate_product_dict(original_dict, product_dict):
    
    if not original_dict or not product_dict:
        return {}

    keys = product_dict.keys()
    values = product_dict.values()
 
    array_product = [{name: dato for name,dato in zip(keys, datos)} for datos in itertools.product(*values)]
    
    return [dict(original_dict, **current_dict) for current_dict in array_product]

