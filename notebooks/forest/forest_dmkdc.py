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



setting = {
    "z_name_of_experiment": 'learning-with-density-matrices',
    "z_run_name": "dmkdc",
    "z_n_components": 1000,
    "z_step": "train_val",
    "z_batch_size": 8,
    "z_dataset": "forest",
    "z_test_running_times": 10
}

#prod_settings = {"z_gamma" : [2**i for i in range(-10,10)], "z_C": [2**i for i in range(-10,10)]}
prod_settings = {"z_gamma" : [2**2]}

params_int = ["z_n_components", "z_batch_size"]
params_float = ["z_gamma"]


# !pwd
mlflow = mlflow_create_experiment(setting["z_name_of_experiment"])

experiments(setting, prod_settings, params_int, params_float, mlflow)
