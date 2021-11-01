# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + [markdown] id="SGS_NkS5zxZX"
# Before running the next cell please add a shortcut to the shared folder at the root of your Google Drive

# + jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
# %load_ext autoreload
# %autoreload 1

# + jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
current_path = ""

try:
  import google.colab
  IN_COLAB = True
except:
  IN_COLAB = False

if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    # %cd /content/drive/MyDrive/Academico/doctorado_programacion/doctorado/experiments/2021_01_learning_with_density_matrices

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

# !pwd

# + jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
import mlflow
import mlflow.sklearn



mlflow.set_tracking_uri("sqlite:///mlflow/tracking.db")
mlflow.set_registry_uri("sqlite:///mlflow/registry.db")
try:
  mlflow.create_experiment('learining-with-density-matrices', "mlflow/")
except:
  print("Experiment already created")
mlflow.set_experiment("learining-with-density-matrices")

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
sys.path.append('scripts/')

# + id="vbwEQAAITCkp"
from load_usps import load_usps

# + jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
print(sys.path)

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 456, "status": "ok", "timestamp": 1613169018839, "user": {"displayName": "sisyphus midas", "photoUrl": "", "userId": "13431807809642753002"}, "user_tz": 300} id="FbvKAN7IyuGq" outputId="f47a6541-41fa-447a-9f93-9a4297d9d362"
X_train, y_train, X_test, y_test = load_usps("data/usps/usps.h5")

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
    "z_n_components": 1000,
    "z_c": 2**1,
    "z_tol": 1e-05,
    "z_experiment": "linear_svc"
}

prod_settings = {"z_gamma" : [2**i for i in range(-10,10)]}

from generate_product_dict import generate_product_dict, add_random_state_to_dict

settings = generate_product_dict(setting, prod_settings)
settings = add_random_state_to_dict(settings)

from experiment_linear_svc import experiment_linear_svc

from scripts.product_dict import generate_product_dict
experiment_linear_svc(X_train, y_train, X_val, y_val, settings, mlflow)

