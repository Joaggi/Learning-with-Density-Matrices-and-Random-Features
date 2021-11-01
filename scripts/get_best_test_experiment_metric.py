from mlflow.entities import ViewType

def get_best_test_experiment_metric(mlflow, experiment_ids, query, metric_to_select):
 
    runs = mlflow.search_runs(experiment_ids=experiment_ids, filter_string=query, run_view_type=ViewType.ACTIVE_ONLY, output_format="pandas")

    best_result = runs[metric_to_select]

    return best_result.mean(), best_result.std()


