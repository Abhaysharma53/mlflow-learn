# inference

import mlflow.pyfunc
import numpy as np

mlflow.set_tracking_uri('http://127.0.0.1:8000')
data = np.array([1,85,66,29,0,26.6,0.351,31]).reshape(1,-1)

model_name = "rf_diabetes"
model_version = 1

model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")

print(model.predict(data))