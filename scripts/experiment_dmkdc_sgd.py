from calculate_metrics import calculate_metrics
import qmc.tf.layers as layers
import qmc.tf.models as models
import numpy as np
import datetime, os


def experiment_dmkdc(X_train, y_train, X_test, y_test, settings, mlflow):
    import mlflow.tensorflow
    mlflow.tensorflow.autolog(every_n_iter=2)

    num_clases = np.unique(y_train).shape[0]

    for setting in settings:

        with mlflow.start_run(run_name=setting["z_run_name"]):
            tensorboard_callback = tf.keras.callbacks.TensorBoard("./mlflow/tensorboard", histogram_freq=1)

            model_mnist = create_model(setting["z_input_dim"], X_train.shape[1], component_dim=setting["z_n_components"], \
                  gamma=setting["z_gamma"], lr=setting["z_learning_rate"], decay=setting["z_decay"], \
                  random_state=setting["z_random_state"], eig_percentage=setting["z_eig_components"], \
                  initialize_with_rff=setting["z_initialize_with_rff"], type_of_rff="rff", fix_rff=False)
            model = models.DMKDClassifier(fm_x=fm_x, dim_x=setting["z_n_components"], num_classes=num_clases)
            model.compile()
            model.fit(X_train, y_train, epochs=1, batch_size=setting["z_batch_size"], verbose=0 )
            
            y_test_pred = model.predict(X_test).argmax(axis=1)
            
            metrics = calculate_metrics(y_test, y_test_pred)

            mlflow.log_params(setting)
            mlflow.log_metrics(metrics)
