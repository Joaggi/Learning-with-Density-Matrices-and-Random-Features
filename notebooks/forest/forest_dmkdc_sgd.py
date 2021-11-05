current_path = ""


try:  
    import google.colab
    IN_COLAB = True
except:
    IN_COLAB = False

if IN_COLAB:
    import os
    os.system("pip3 install mlflow")

    from google.colab import drive
    drive.mount('/content/drive') os.chdir('/content/drive/MyDrive/Academico/doctorado_programacion/experiments/2021_01_learning_with_density_matrices') import sys sys.path.append('submodules/qmc/') #sys.path.append('../../../../submodules/qmc/')
    print(sys.path)
else:
    import sys
    sys.path.append('submodules/qmc/')
    sys.path.append('data/')
    #sys.path.append('../../../../submodules/qmc/')
    print(sys.path)
    # %cd ../../

print(os.getcwd())

sys.path.append('scripts/')


import qmc.tf.layers as layers
import qmc.tf.models as models

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

import sklearn
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV, KFold

from experiments import experiments

from mlflow_create_experiment import mlflow_create_experiment
    "z_name_of_experiment": 'learning-with-density-matrices',
    "z_n_components": 1000,

setting = {
    "z_run_name": "dmkdc_sgd",
    "z_step": "train_val",
    "z_batch_size": 8,
    "z_learning_rate": 1e-06,
    "z_decay": 0.0,
    "z_initialize_with_rff": True,
    "z_type_of_rff": "rff",
    "z_fix_rff": True, 
    "z_train_epochs": 50, 
    "z_test_epochs": 2000, 
    "z_dataset": "forest",
    "z_test_running_times": 10,
    "z_random_search": True,
    "z_random_search_iter": 30,
    "z_random_search_random_state": 20


}

prod_settings = {"z_gamma" : [2**i for i in range(-20,20)], "z_eig_components": [0.0, 0.1, 0.5]}
#prod_settings = {"z_gamma" : [2**2], "z_eig_components": [0.1]}

params_int = ["z_n_components", "z_batch_size", "z_epochs"]
params_float = ["z_gamma", "z_eig_components", "z_learning_rate", "z_decay"]
params_boolean = ["z_initialize_with_rff", "z_fix_rff"]

mlflow = mlflow_create_experiment(setting["z_name_of_experiment"])

experiments(setting, prod_settings, params_int, params_float, mlflow)
