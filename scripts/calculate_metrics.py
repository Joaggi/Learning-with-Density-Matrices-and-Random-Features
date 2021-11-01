from sklearn.metrics import accuracy_score

def calculate_metrics(y_true, y_pred):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred)
    }
    return metrics
