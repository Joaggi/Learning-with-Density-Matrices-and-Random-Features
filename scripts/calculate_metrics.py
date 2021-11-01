def calculate_metrics(y_true, y_pred):
    metrics = {
        "accuracy":model.score(y_true, y_pred)
    }
    return metrics
