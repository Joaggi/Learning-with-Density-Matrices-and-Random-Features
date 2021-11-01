---
jupyter:
  jupytext:
    encoding: '# -*- coding: utf-8 -*-'
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.0
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region id="SGS_NkS5zxZX" -->
Before running the next cell please add a shortcut to the shared folder at the root of your Google Drive
<!-- #endregion -->

```python jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
%load_ext autoreload
%autoreload 1
```

```python jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
current_path = ""

try:
  import google.colab
  IN_COLAB = True
except:
  IN_COLAB = False

if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    %cd /content/drive/MyDrive/Academico/doctorado_programacion/doctorado/experiments/2021_01_learning_with_density_matrices

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
    %cd ../../

!pwd
```

```python jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("sqlite:///mlflow/tracking.db")
mlflow.set_registry_uri("sqlite:///mlflow/registry.db")
try:
  mlflow.create_experiment('learining-with-density-matrices', "mlflow/")
except:
  print("Experiment already created")
mlflow.set_experiment("learining-with-density-matrices")
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 15976, "status": "ok", "timestamp": 1613168987500, "user": {"displayName": "sisyphus midas", "photoUrl": "", "userId": "13431807809642753002"}, "user_tz": 300} id="xN741Hz3S2Gw" outputId="591d1f6b-ab20-4021-aa06-3cfef2daf887"
import qmc.tf.layers as layers
import qmc.tf.models as models

```


```python id="3gmovb6m0FFl"
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
```

```python id="ri9fmngn_GPA"
import sklearn
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV, KFold
```

```python id="DZbvtfnXQD9Y"
# from functions.keras_wrapper_qmc import KerasClassifier

# from sklearn.model_selection import RandomizedSearchCV, KFold
# from sklearn.metrics import make_scorer
```

```python jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
sys.path.append('scripts/')
```

```python id="vbwEQAAITCkp"
from load_usps import load_usps
```

```python jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
print(sys.path)
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 456, "status": "ok", "timestamp": 1613169018839, "user": {"displayName": "sisyphus midas", "photoUrl": "", "userId": "13431807809642753002"}, "user_tz": 300} id="FbvKAN7IyuGq" outputId="f47a6541-41fa-447a-9f93-9a4297d9d362"
X_train, y_train, X_test, y_test = load_usps("data/usps/usps.h5")

print("shape X_train : ", X_train.shape)
print("shape y_train : ", y_train.shape)
print("shape X_test : ", X_test.shape)
print("shape y_test : ", y_test.shape)
```

```python id="CCp5RUItvNfu"

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
```

```python id="q5sHceZTvA4W"
from min_max_scaler import min_max_scaler
X_train, X_val, X_test = min_max_scaler(X_train, X_val, X_test)
```

```python jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)
print(X_test.shape)
print(y_test.shape)
```


```python id="sbBAEyWtvCQs"
setting = {
    "n_components": 1000,
    "c": 2**1,
    "tol": 1e-05
}

gammas = [2**i for i in range(-10,10)]
```


```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 3953, "status": "ok", "timestamp": 1613169028113, "user": {"displayName": "sisyphus midas", "photoUrl": "", "userId": "13431807809642753002"}, "user_tz": 300} id="xC2A4DDR2ke2" outputId="bc4e23b0-e98a-4940-863c-edaf52d251bf"
from rbf_sampler import rbf_sampler
X_train_features, X_val_features, X_test_features = rbf_sampler( X_train, X_val, X_test, gamma=2**-5, n_components = 1000, random_state=42)
```

```python jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
from linear_svc import linear_svc
linear_svc(X_train_features, y_train, X_val, y_val, C=2**1, tol=1e-05)
```


```python id="WJhIK0PPs0ZC"
input_dim = X_train.shape[1]
component_dim = 1000
num_classes = np.unique(y_train).shape[0]
gamma = 2**-5
random_state=0
num_eig=0
batch_size=8
epochs = 10
```

```python id="8lkO0ZHm7mGQ"
def create_model(input_dim, num_classes, component_dim=100, gamma=1, lr=0.01, decay=0.,
                  random_state=None, eig_percentage=0, initialize_with_rff=False,
                  type_of_rff="rff", fix_rff=False):
    '''This is a model generating function so that we can search over neural net
    parameters and architecture'''

    num_eig = round(eig_percentage * component_dim)

    opt = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, decay=decay)

    if type_of_rff == 'rff':
        fm_x = layers.QFeatureMapRFF(input_dim, dim=component_dim, gamma=gamma, random_state=random_state)
    else:
        fm_x = layers.QFeatureMapORF(input_dim, dim=component_dim, gamma=gamma, random_state=random_state)

    if initialize_with_rff:
        qmkdc = models.QMKDClassifier(fm_x=fm_x, dim_x=component_dim, num_classes=num_classes)
        qmkdc.compile()
        qmkdc.fit(X_train, y_train, epochs=1, batch_size=batch_size, verbose=0)

    qmkdc1 = models.QMKDClassifierSGD(input_dim=input_dim, dim_x=component_dim, num_eig=num_eig,
                                      num_classes=num_classes, gamma=gamma, random_state=random_state, fm_x=fm_x)

    if fix_rff:
        qmkdc1.layers[0].trainable = False

    qmkdc1.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    if initialize_with_rff:
        qmkdc1.set_rhos(qmkdc.get_rhos())

    # qmkdc1.fit(X_train, y_train_bin, epochs=epochs, batch_size=batch_size)

    return qmkdc1
```

```python id="6aAqByT7EqwD"
%load_ext tensorboard
%tensorboard --logdir "/gdrive/My Drive/logs/"
```

```python colab={"base_uri": "https://localhost:8080/"} id="iA1l5hucdWnb" outputId="8b42e155-031c-409c-e49b-51fe271db0b4"
# Creación del modelo sin random search. Sirve para verificar que el algoritmo este entrenando bien.
import datetime, os
logdir = os.path.join("/gdrive/My Drive/logs", "usps-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

model_mnist = create_model(input_dim, num_classes, component_dim=1000, gamma=2**-6, lr=1e-06, decay=0.,
                  random_state=None, eig_percentage=0.1, initialize_with_rff=True,
                  type_of_rff="rff", fix_rff=False)

y_train_bin = tf.reshape(tf.keras.backend.one_hot(y_train, num_classes), (-1, num_classes))
y_test_bin = tf.reshape(tf.keras.backend.one_hot(y_test, num_classes), (-1, num_classes))
model_mnist.fit(X_train, y_train_bin.numpy(), epochs=12000, batch_size=32, validation_data=(X_test, y_test_bin.numpy()), 
            callbacks=[tensorboard_callback])
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1703, "status": "ok", "timestamp": 1606083504277, "user": {"displayName": "Joseph Alejandro Gallego Mejia", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiRHNpVI1SA3eB-WTWWN6LPuOQg9GlITBmVJTaXloc=s64", "userId": "11469130437763358793"}, "user_tz": 300} id="CO-lHE9-mK9i" outputId="7d193189-b405-4281-ac7d-2f602e068146"
y_test_bin = tf.reshape(tf.keras.backend.one_hot(y_test, num_classes), (-1, num_classes))
out = model_mnist.predict(X_test)
accuracy_score(y_test, out.argmax(axis=1))
```

```python id="el95YvDq7xm_"
model = KerasClassifier(build_fn=create_model, epochs=50, batch_size=16, verbose=1)
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 297, "status": "ok", "timestamp": 1606083365928, "user": {"displayName": "Joseph Alejandro Gallego Mejia", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiRHNpVI1SA3eB-WTWWN6LPuOQg9GlITBmVJTaXloc=s64", "userId": "11469130437763358793"}, "user_tz": 300} id="4UPoG_17pLhW" outputId="8c09a56b-82ab-49c5-f5c5-c25c8c44360d"
num_classes
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 454, "status": "ok", "timestamp": 1606083366868, "user": {"displayName": "Joseph Alejandro Gallego Mejia", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiRHNpVI1SA3eB-WTWWN6LPuOQg9GlITBmVJTaXloc=s64", "userId": "11469130437763358793"}, "user_tz": 300} id="JOwKE_ARyc9Q" outputId="b933e3b7-3111-46f5-dae8-839cd746baf6"
2**30
```

```python id="6YtfnhVn70Oz"
# component dimension
from scipy.stats import randint
#components_dimensions = randint(20,1500)
components_dimensions = [1000]

# gamma
#gammas = [2 ** i for i in range(-25, 25)]
gammas = [2 ** i for i in range(4, 6)]

# number of eigen values
eig_values = [1 / 10, 25 / 100, 50 / 100, 1]

# initialize with rff
#initialize_with_rff = [True, False]
initialize_with_rff = [True]

# type_of_rff_components
type_of_rff = ['rff', 'orf']

# fix_rff
fix_rff = [True]

# learning algorithm parameters
lr = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
decay = [1e-6, 1e-9, 0]

# dictionary summary
param_grid = dict(input_dim=[input_dim], num_classes=[num_classes], component_dim=components_dimensions, gamma=gammas,
                  lr=lr, decay=decay,
                  random_state=[None], eig_percentage=eig_values, initialize_with_rff=initialize_with_rff,
                  type_of_rff=type_of_rff, fix_rff=fix_rff
                  )
```

```python id="qtgVrnlQ8HM_"
grid = RandomizedSearchCV(estimator=model, cv=KFold(3), param_distributions=param_grid,
                              verbose=20, n_iter=1, n_jobs=1)
```

```python id="7bcv-TfgL3tU"
# grid_result = grid.fit(X_train, y_train)
y_train_bin = tf.reshape(tf.keras.backend.one_hot(y_train, num_classes), (-1, num_classes)).numpy()
```

```python id="UI63v1ne8VFi"
cv_results_df = None

for i in range(25):
  grid_result = grid.fit(X_train, y_train_bin)

  if cv_results_df is None:
    cv_results_df = pd.DataFrame(grid_result.cv_results_)
  else:
    cv_results_df = pd.concat([cv_results_df, pd.DataFrame(grid_result.cv_results_)])

  cv_results_df.to_csv('qmkdc letters random search.csv')
```
