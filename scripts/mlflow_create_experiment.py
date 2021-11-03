import mlflow
import mlflow.sklearn



def mlflow_create_experiment(name_of_experiment):

    mlflow.set_tracking_uri("sqlite:///mlflow/tracking.db")
    mlflow.set_registry_uri("sqlite:///mlflow/registry.db")
    try:
      mlflow.create_experiment(name_of_experiment, "mlflow/")
    except:
      print("Experiment already created")
    mlflow.set_experiment(name_of_experiment)

    return mlflow
