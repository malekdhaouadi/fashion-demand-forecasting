   data:
   	python -m src.data.load
   train:
      python -m src.models.baseline cluster_style0.3 series

   mlflow-ui:
   mlflow ui --backend-store-uri sqlite:///mlflow.db