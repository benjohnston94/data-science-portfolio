#%%
# General
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

# Modelling
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

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

    # preprocessor
    scaler = StandardScaler()

    # Hyper parameter search

    # Models

    # currently unused
    # Naives Bayes
    # nb_clf = GaussianNB()
    # Logistic regresion
    # lr_clf = LogisticRegression()

    # Random forest
    rf_clf = RandomForestClassifier()

    rf_classifier = Pipeline(
        [
            ("scl", scaler),
            ("rf", rf_clf)
        ]
    )

    rf_param_grid = {
        "rf__n_estimators": [50, 100],
        "rf__criterion": ["gini", "entropy"],
        "rf__max_depth": [1, 3, 5]
    }

    rkfold_cv = RepeatedKFold(n_splits=5, n_repeats=5)

    rf_search = GridSearchCV(
        estimator=rf_classifier,
        param_grid=rf_param_grid,
        scoring="f1",
        verbose=10,
        cv=rkfold_cv,
        return_train_score=True
    )

    rf_search.fit(X_train, y_train)

    rf_model = rf_search.best_estimator_["rf"]

    print_model_scores(rf_search)

    # Evaluating models

    # rf_coefs = pd.DataFrame(
    #     rf_model.feature_importances_,
    #     columns=["Coefficients"],
    #     index=X.columns
    # )
    
    # Feature importance

    feature_names = X.columns
    feature_importances = rf_model.feature_importances_
    sorted_idx = feature_importances.argsort()

    [print(f"{name}: {value}") for name, value in zip(feature_names[sorted_idx], feature_importances[sorted_idx])]
    
    fig = plt.figure(figsize=(10,5))
    plt.barh(feature_names[sorted_idx], feature_importances[sorted_idx])
    # fig.savefig("feature importance.png")
    plt.show()
