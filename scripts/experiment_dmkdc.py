from calculate_metrics import calculate_metrics
import qmc.tf.layers as layers
import qmc.tf.models as models
import numpy as np

def experiment_dmkdc(X_train, y_train, X_test, y_test, settings, mlflow):
    num_clases = np.unique(y_train).shape[0]

    for i, setting in enumerate(settings):
        print(f"experiment_dmkdc {i}")
        with mlflow.start_run(run_name=setting["z_run_name"]):
            fm_x = layers.QFeatureMapRFF(X_train.shape[1], dim=setting["z_n_components"], gamma=setting["z_gamma"], random_state=setting["z_random_state"])
        
            model = models.DMKDClassifier(fm_x=fm_x, dim_x=setting["z_n_components"], num_classes=num_clases)
            model.compile()
            model.fit(X_train, y_train, epochs=1, batch_size=setting["z_batch_size"], verbose=0 )
            
            y_test_pred = model.predict(X_test).argmax(axis=1)
            
            metrics = calculate_metrics(y_test, y_test_pred)

            mlflow.log_params(setting)
            mlflow.log_metrics(metrics)
