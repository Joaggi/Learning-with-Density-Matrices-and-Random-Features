from experiment_linear_svc import experiment_linear_svc
from experiment_dmkdc_sgd import experiment_dmkdc_sgd
from experiment_dmkdc import experiment_dmkdc


def make_experiment(algorithm, X_train, y_train, X_val, y_val, settings, mlflow):
    
    if algorithm == "linear_svc":
        experiment_linear_svc(X_train, y_train, X_val, y_val, settings, mlflow)

    if algorithm == "dmkdc":
        experiment_dmkdc(X_train, y_train, X_val, y_val, settings, mlflow)
 
    if algorithm == "dmkdc_sgd":
        experiment_dmkdc_sgd(X_train, y_train, X_val, y_val, settings, mlflow)
