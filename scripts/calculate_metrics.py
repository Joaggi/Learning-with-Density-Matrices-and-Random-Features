def calculate_metrics(model, X, y):
    metrics = {
        "accuracy":model.score(X, y)
    }
    return metrics
