
def convert_best_train_experiment_to_settings_of_test(best_experiment, settings_int, settings_float):
    best_experiment = dict(best_experiment, **{"z_state": "test"})
    for param in settings_int:
        best_experiment[param] = int(best_experiment[param])
    for param in settings_float:
        best_experiment[param] = float(best_experiment[param])

    return best_experiment


