from mlflow.tracking.client import MlflowClient
from mlflow.entities import ViewType


def get_best_val_experiment(mlflow, experiment_ids, query, metric_to_select):
 
    runs = mlflow.search_runs(experiment_ids=experiment_ids, filter_string=query, run_view_type=ViewType.ACTIVE_ONLY, output_format="pandas")

    best_result = runs.sort_values(metric_to_select, ascending=False).iloc[0]
    print(best_result)

    keys = best_result.keys()
    filter = keys.str.match(r'(^params\.*)')
    best_params = best_result[keys[filter]]
    keys = best_params.keys()
    new_keys = keys.str.replace('params.','')
    best_params.index = new_keys

    return best_params


