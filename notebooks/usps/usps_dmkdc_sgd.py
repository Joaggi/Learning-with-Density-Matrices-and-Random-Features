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
    drive.mount('/content/drive')
    os.chdir('/content/drive/MyDrive/Academico/doctorado_programacion/experiments/2021_01_learning_with_density_matrices')
    import sys
    sys.path.append('submodules/qmc/')
    #sys.path.append('../../../../submodules/qmc/')
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

from mlflow_create_experiment import mlflow_create_experiment
name_of_experiment = 'learning-with-density-matrices'
mlflow = mlflow_create_experiment(name_of_experiment)


# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 15976, "status": "ok", "timestamp": 1613168987500, "user": {"displayName": "sisyphus midas", "photoUrl": "", "userId": "13431807809642753002"}, "user_tz": 300} id="xN741Hz3S2Gw" outputId="591d1f6b-ab20-4021-aa06-3cfef2daf887"
import qmc.tf.layers as layers
import qmc.tf.models as models



# + id="3gmovb6m0FFl"
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# + id="ri9fmngn_GPA"
import sklearn
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV, KFold

# + id="DZbvtfnXQD9Y"
# from functions.keras_wrapper_qmc import KerasClassifier

# from sklearn.model_selection import RandomizedSearchCV, KFold
# from sklearn.metrics import make_scorer

# + jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
print(sys.path)



# + id="vbwEQAAITCkp"
from load_dataset import load_dataset
# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 456, "status": "ok", "timestamp": 1613169018839, "user": {"displayName": "sisyphus midas", "photoUrl": "", "userId": "13431807809642753002"}, "user_tz": 300} id="FbvKAN7IyuGq" outputId="f47a6541-41fa-447a-9f93-9a4297d9d362"
X_train, y_train, X_test, y_test = load_dataset("usps")

print("shape X_train : ", X_train.shape)
print("shape y_train : ", y_train.shape)
print("shape X_test : ", X_test.shape)
print("shape y_test : ", y_test.shape)

# + id="CCp5RUItvNfu"

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

# + id="q5sHceZTvA4W"
from min_max_scaler import min_max_scaler
X_train, X_val, X_test = min_max_scaler(X_train, X_val, X_test)

# + jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)
print(X_test.shape)
print(y_test.shape)


# + id="sbBAEyWtvCQs"
setting = {
    "z_run_name": "dmkdc_sgd",
    "z_n_components": 1000,
    "z_step": "train_val",
    "z_batch_size": 8,
    "z_learning_rate": 1e-06,
    "z_decay": 0.0,
    "z_initialize_with_rff": True,
    "z_type_of_rff": "rff",
    "z_fix_rff": False, 
    "z_epochs": 2000, 
    "z_dataset": "letters"
}

#prod_settings = {"z_gamma" : [2**i for i in range(-10,10)], "z_C": [2**i for i in range(-10,10)]}
prod_settings = {"z_gamma" : [2**-6], "z_eig_components": [0.1]}

params_int = ["z_n_components", "z_batch_size"]
params_float = ["z_gamma", "z_eig_components", "z_learning_rate", "z_decay"]
params_boolean = ["z_initialize_with_rff", "z_fix_rff"]

from generate_product_dict import generate_product_dict, add_random_state_to_dict, generate_several_dict_with_random_state

settings = generate_product_dict(setting, prod_settings)
settings = add_random_state_to_dict(settings)

from experiment_dmkdc_sgd import experiment_dmkdc_sgd

experiment_dmkdc_sgd(X_train, y_train, X_val, y_val, settings, mlflow)


experiments_list = mlflow.get_experiment_by_name(name_of_experiment)
experiment_id = experiments_list.experiment_id
    
from get_best_val_experiment import get_best_val_experiment
query = f"params.z_run_name = '{setting['z_run_name']}' and params.z_step = 'train_val'"
metric_to_evaluate = "metrics.accuracy"
best_experiment = get_best_val_experiment(mlflow, experiment_id,  query, metric_to_evaluate)
from convert_best_train_experiment_to_settings_of_test import convert_best_train_experiment_to_settings_of_test
best_experiment = convert_best_train_experiment_to_settings_of_test(best_experiment, params_int, params_float, params_boolean)

settings_test = generate_several_dict_with_random_state(best_experiment, 10)

experiment_dmkdc_sgd(np.concatenate([X_train, X_val]), \
    np.concatenate([y_train, y_val]), X_test, y_test, settings_test, mlflow)

from get_best_test_experiment_metric import get_best_test_experiment_metric
query = f"params.z_run_name = '{setting['z_run_name']}' and params.z_step = 'test'"
metric_to_evaluate = "metrics.accuracy"
get_best_test_experiment_metric(mlflow, experiment_id,  query, metric_to_evaluate)
