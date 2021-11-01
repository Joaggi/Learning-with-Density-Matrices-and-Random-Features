#!/bin/bash

mlflow ui --backend-store-uri sqlite:///mlflow/tracking.db --port 5000 &
