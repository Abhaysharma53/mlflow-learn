import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import mlflow

data = pd.read_csv('https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv')

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)

rf = RandomForestClassifier(random_state= 42)

param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [2, 3, 5, 7]
}

#applying grid search
grid_search = GridSearchCV(estimator= rf, param_grid= param_grid, cv= 5, n_jobs= -1, verbose= True)

mlflow.set_experiment('diabetes-rf-hp')

with mlflow.start_run(description = 'Best hyperparameter trained RF model'):
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    mlflow.log_params(best_params)
    mlflow.log_metric("accuracy", best_score)
    mlflow.log_artifact(__file__)

    #infer model signature
    signature = mlflow.models.infer_signature(X_train, grid_search.best_estimator_.predict(X_train))
    mlflow.sklearn.log_model(grid_search.best_estimator_, "random forest", signature = signature)


    #track data - first convert to mlflow numpy array
    train_data = X_train
    train_data['Outcome'] = y_train

    test_data = y_train
    test_data['Outcome'] = y_test

    #conversion of data to mlflow accepted format
    train_data = mlflow.data.from_pandas(train_data)
    test_data = mlflow.data.from_pandas(test_data)

    #tracking
    mlflow.log_input(train_data, context = 'Training')
    mlflow.log_input(test_data, context = 'Validation')

    #setting tags

    mlflow.set_tag("Author", "Abhay")

    print(best_params)
    print(best_score)



    



