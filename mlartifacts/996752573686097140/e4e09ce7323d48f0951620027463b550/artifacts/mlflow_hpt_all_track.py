import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import mlflow

data = pd.read_csv('https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv')

mlflow.set_tracking_uri('http://127.0.0.1:8000')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)

et = ExtraTreesClassifier(random_state= 42)

param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [2, 3, 5, 7, 8, 10]
}

#applying grid search
grid_search = GridSearchCV(estimator= et, param_grid= param_grid, cv= 5, n_jobs= -1, verbose= True)

mlflow.set_experiment('diabetes-etc-hypertune-all track')

with mlflow.start_run(description = 'Best hyperparameter trained ETC model') as parent:
    grid_search.fit(X_train, y_train)


    for i in range(len(grid_search.cv_results_['params'])):
        print(i)
        with mlflow.start_run(nested = True) as child:
            mlflow.log_params(grid_search.cv_results_['params'][i])
            mlflow.log_metric("accuracy", grid_search.cv_results_['mean_test_score'][i])

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    mlflow.log_params(best_params)
    mlflow.log_metric("accuracy", best_score)
    mlflow.log_artifact(__file__)

    #infer model signature
    signature = mlflow.models.infer_signature(X_train, grid_search.best_estimator_.predict(X_train))
    mlflow.sklearn.log_model(grid_search.best_estimator_, "Extra Tree classifier", signature = signature)


    #track data - first convert to mlflow numpy array
    train_data = X_train
    train_data['Outcome'] = y_train

    test_data = X_test
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



    



