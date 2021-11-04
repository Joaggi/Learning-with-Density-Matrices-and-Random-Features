from calculate_metrics import calculate_metrics
import qmc.tf.layers as layers
import qmc.tf.models as models
import numpy as np
import datetime, os

import tensorflow as tf

from dmkdc_sgd import create_model

def experiment_dmkdc_sgd(X_train, y_train, X_test, y_test, settings, mlflow):
    #import mlflow.tensorflow
    #mlflow.tensorflow.autolog(every_n_iter=2)

    num_classes = np.unique(y_train).shape[0]

    for setting in settings:

        with mlflow.start_run(run_name=setting["z_run_name"]):
            tensorboard_callback = tf.keras.callbacks.TensorBoard("./mlflow/tensorboard", histogram_freq=1)

            model = create_model(X_train, y_train, X_train.shape[1], num_classes, component_dim=setting["z_n_components"], \
                  gamma=setting["z_gamma"], lr=setting["z_learning_rate"], decay=setting["z_decay"], \
                  random_state=setting["z_random_state"], eig_percentage=setting["z_eig_components"], \
                  initialize_with_rff=setting["z_initialize_with_rff"], type_of_rff=setting["z_type_of_rff"], \
                  fix_rff=setting["z_fix_rff"], batch_size = setting["z_batch_size"]) 
       
            y_train_bin = tf.reshape(tf.keras.backend.one_hot(y_train, num_classes), (-1, num_classes))
            y_test_bin = tf.reshape(tf.keras.backend.one_hot(y_test, num_classes), (-1, num_classes))
            model.fit(X_train, y_train_bin.numpy(), epochs=setting["epochs"], batch_size=setting["z_batch_size"], \
                validation_data=(X_test, y_test_bin.numpy()), callbacks=[tensorboard_callback])
            y_test_bin = tf.reshape(tf.keras.backend.one_hot(y_test, num_classes), (-1, num_classes))
            out = model.predict(X_test)
            
            y_test_pred = out.argmax(axis=1)
           
            metrics = calculate_metrics(y_test, y_test_pred)

            mlflow.log_params(setting)
            mlflow.log_metrics(metrics)
