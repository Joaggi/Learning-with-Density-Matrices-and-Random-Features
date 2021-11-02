from linear_svc import linear_svc
from rbf_sampler import rbf_sampler
from linear_svc import linear_svc
from calculate_metrics import calculate_metrics

def experiment_linear_svc(X_train, y_train, X_test, y_test, settings, mlflow):
    
    for setting in settings:

        with mlflow.start_run(run_name=setting["z_run_name"]):
            X_train_features, X_test_features = \
                rbf_sampler( X_train, X_test, gamma=setting["z_gamma"], n_components = setting["z_n_components"], \
                random_state=setting["z_random_state"])

            model = linear_svc(X_train_features, y_train, C=setting["z_C"], tol=setting["z_tol"], max_iter=setting["z_max_iter"])
        
            y_test_pred = model.predict(X_test_features)
            metrics = calculate_metrics(y_test, y_test_pred)

            mlflow.log_params(setting)
            mlflow.log_metrics(metrics)
