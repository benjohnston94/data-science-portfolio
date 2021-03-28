#%%
### Practicing different cross validation methods

# General
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

# Modelling
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    GridSearchCV,
    RepeatedKFold,
    StratifiedKFold
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

# Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Constants
TEST_SIZE = 0.3
RANDOM_STATE = 101

# functions
def print_model_scores(model):
# def print_model_scores(model, X_train=X_train, X_test=X_test): # don't think I need these

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    
    # print(f"accuracy: {accuracy_score(y_test, predictions)}") 
    print(f"train precision: {precision_score(y_train, train_predictions)}  | test precision: {precision_score(y_test, test_predictions)}")
    print(f"train recall: {recall_score(y_train, train_predictions)}    | test recall: {recall_score(y_test, test_predictions)}")
    print(f"train f1 score: {f1_score(y_train, train_predictions)}  | test f1 score: {f1_score(y_test, test_predictions)}  |")

if __name__=="__main__":
    # read in data
    df = pd.read_pickle("titanic_train_clean")
    X = df.drop(["Survived"], axis=1)
    y = df["Survived"]

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    """
    Want to build a logistic regression model but in a general enough way (say in terms of cross validation)
    that this code structure can be used for a variety of models
    """

    # scaler
    scaler = StandardScaler()
    
    # CV
    repeated_kfold = RepeatedKFold(n_splits=5, n_repeats=5)
    stratified_kfold_no_shuffle = StratifiedKFold()
    stratified_kfold_shuffle = StratifiedKFold(shuffle=True)

    cv_list = [repeated_kfold, stratified_kfold_no_shuffle, stratified_kfold_shuffle]

    # using built in
    # def run_cvs(run_switch=False):
    # if run_switch:
    for i, cv in enumerate(cv_list):
        logistic_reg_model = LogisticRegressionCV(cv=cv, verbose=10)

        pipe = Pipeline(
            [
                ("scaler", scaler),
                ("lr", logistic_reg_model)
            ]
        )

        pipe.fit(X_train, y_train)

        print(i)
        print_model_scores(pipe)

    # run_cvs(run_switch=True)

    print(pipe["lr"].coef_)
