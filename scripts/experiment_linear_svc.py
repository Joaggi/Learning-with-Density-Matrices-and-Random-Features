from .linear_svc import linear_svc

def experiment_linear_svc(settings, mlflow):
    
    for setting in settings:
        model = linear_svc(X_train_features, y_train, C=setting["z_C"], tol=setting["z_tol"])
    
        y_val = model.predict(X_val)
        metrics = calculate_metrics(test_true_density, estimated_density)

        mlflow.log_params(setting)
        mlflow.log_metrics(metrics)


