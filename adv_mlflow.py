import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

#data = pd.read_csv('https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv')

# X = data.iloc[:, :-1]
# y = data.iloc[:, 1]
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameters for the Random Forest model
max_depth = 2
n_estimators = 50

# apply mlflow
mlflow.set_experiment('iris-rf')

with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth)

    rf.fit(X_train, y_train)


    y_pred = rf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    #log metric
    mlflow.log_metric('accuracy', accuracy)

    #log params
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('n_estimators', n_estimators)

    # Create a confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    #plt.show()
    
    # Save the plot as an artifact
    plt.savefig("confusion_matrix.png")

#   # log artifact
    mlflow.log_artifact("confusion_matrix.png")

    mlflow.log_artifact(__file__)

    mlflow.sklearn.log_model(rf, "Random forest")

    mlflow.set_tag('author','Abhay')
    mlflow.set_tag('model','random forest')

    #logging dataset
    # train_df = X_train
    # train_df['species'] = int(y_train)

    # test_df  = X_test
    # test_df['species'] = int(y_test)

    # #convert pandas dataframe to mlflow numpy array
    # train_df = mlflow.data.from_pandas(train_df)
    # test_df = mlflow.data.from_pandas(test_df)

    # mlflow.log_input(train_df, context = 'training')
    # mlflow.log_input(test_df, context = 'test')

    print('accuracy', accuracy)
    print(y_train)