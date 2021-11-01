from linear_svc import linear_svc
from rbf_sampler import rbf_sampler
from linear_svc import linear_svc
from calculate_metrics import calculate_metrics

def experiment_linear_svc(X_train, y_train, X_val, y_val, settings, mlflow):
    
    for setting in settings:
        X_train_features, X_val_features = \
            rbf_sampler( X_train, X_val, gamma=setting["z_gamma"], n_components = setting["z_n_components"], \
            random_state=setting["z_random_state"])

        model = linear_svc(X_train, y_train, C=setting["z_C"], tol=setting["z_tol"])
    
        y_val_pred = model.predict(X_val)
        metrics = calculate_metrics(y_val, y_val_pred)

        mlflow.log_params(setting)
        mlflow.log_metrics(metrics)

