from load_dataset import load_dataset
from min_max_scaler import min_max_scaler
from sklearn.model_selection import train_test_split
from generate_product_dict import generate_product_dict, add_random_state_to_dict, generate_several_dict_with_random_state
from get_best_val_experiment import get_best_val_experiment
from convert_best_train_experiment_to_settings_of_test import convert_best_train_experiment_to_settings_of_test
from get_best_test_experiment_metric import get_best_test_experiment_metric
from make_experiment import make_experiment
import numpy as np

def experiments(setting, prod_settings, params_int, params_float, mlflow):

    algorithm = setting["z_run_name"]
    dataset = setting["z_dataset"]
    name_of_experiment = setting["z_name_of_experiment"]

    X_train, y_train, X_test, y_test = load_dataset(dataset)

    print("shape X_train : ", X_train.shape)
    print("shape y_train : ", y_train.shape)
    print("shape X_test : ", X_test.shape)
    print("shape y_test : ", y_test.shape)



    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

    X_train, X_val, X_test = min_max_scaler(X_train, X_val, X_test)

    print(X_train.shape)
    print(y_train.shape)
    print(X_val.shape)
    print(y_val.shape)
    print(X_test.shape)
    print(y_test.shape)

    if "z_train_val_enabled" in setting and setting["z_train_val_enabled"] == True:

        settings = generate_product_dict(setting, prod_settings)
        settings = add_random_state_to_dict(settings)

        make_experiment(algorithm, X_train, y_train, X_val, y_val, settings, mlflow)

    

    experiments_list = mlflow.get_experiment_by_name(name_of_experiment)
    experiment_id = experiments_list.experiment_id

    if "z_select_best_val_experiment" in setting and setting["z_select_best_val_experiment"] == True:

        query = f"params.z_run_name = '{setting['z_run_name']}' and params.z_step = 'train_val' and params.z_dataset = '{setting['z_dataset']}'"
        metric_to_evaluate = "metrics.accuracy"
        best_experiment = get_best_val_experiment(mlflow, experiment_id,  query, metric_to_evaluate)
        best_experiment = convert_best_train_experiment_to_settings_of_test(best_experiment, params_int, params_float)

        settings_test = generate_several_dict_with_random_state(best_experiment, setting["z_test_running_times"])

    else:
        settings_test = generate_product_dict(setting, prod_settings)
        settings_test = add_random_state_to_dict(settings_test)



    make_experiment(algorithm, np.concatenate([X_train, X_val]), \
        np.concatenate([y_train, y_val]), X_test, y_test, settings_test, mlflow)

    query = f"params.z_run_name = '{setting['z_run_name']}' and params.z_step = 'test' and params.z_dataset = '{setting['z_dataset']}'"
    metric_to_evaluate = "metrics.accuracy"
    get_best_test_experiment_metric(mlflow, experiment_id,  query, metric_to_evaluate)
