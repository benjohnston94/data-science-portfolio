#%%
"""
Modelling script

Notes on imbalance target class:
* Accuracy won't be a good measure so should focus on optimizing
  precision or recall. Because we're looking at loan defaults, false positives are likely
  better than false negatives (it's bad if we *miss* a customer who may default)
  Thus, we'll use either use recall or f-beta score as the main metric to measure
* When training we'll need to stratify the target in the cross validation loop

Model selection:
For this kind of classification models we could use include a Naive Bayes classifiear, 
logistic regression, random forest, support vector machines etc.
We'll try a few models, but in theory, a linear model should make sense as we have 
relatively few features compared to the sample size (n > p). 

Notes on logistic regression:
* Starting with a linear model will serve as a good benchmark, and has the advantage of interpretability
* We'll need some regularisation so we don't overfit. Can tune:
    * regularisation method (l2, l1, elastic-net)
    * Strength of regularisation (C)
* In order for coefficients to be comparable, we'll need to scale the input data
* Another consideration is that logistic regression gives a probabilistic output,
  which for this use-case is probably a helpful output. For example, if communicating to stakeholders,
  or visualising this score in a dashboard, a probabilistic score is easy to interpret. Though we have a
  'model threshold' for predictions, a probabilistic score could also potentially be used for different
  courses of action for different thresholds. (Say, different types of intervention)

Steps:
* Imports
* Constants
* Pipeline:
    * (Potentially feature engineering - say, apply polynomial function)
    * Scale data
    * Model
* Grid-search for parameters with stratified cross validation
"""

# Imports - standard
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib import rcParams              # this is for ensuring formatting looks okay
rcParams.update({'figure.autolayout': True}) # when we save any matplotlib figures

# Imports - modelling
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score
)
from sklearn.model_selection import (
    train_test_split,
    RepeatedStratifiedKFold,
    GridSearchCV
)
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.pipeline import Pipeline

### Constants ###
TEST_SIZE = 0.3
RANDOM_STATE = 101
TARGET_COL = ["Credit Default"] # need to update pipeline to change to snakecase
BETA = 2 # used for f-beta scores; weighs recalls higher than precision 


### Data ###
df = pd.read_pickle("data_cleaned.pkl")

X = df.drop(TARGET_COL, axis=1)
y = df[TARGET_COL]

X_array = X.values
y_array = y.values.ravel()

X_train, X_test, y_train, y_test = train_test_split(
    X_array,
    y_array,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)


### Modelling ###
pipe = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("regressor", LogisticRegression())
    ]
)

param_grid = [
    {
        "regressor__C": np.logspace(-3, 0, 4),
        "regressor__solver": ["liblinear"],
        "regressor__penalty": ["l1"]
    },
    {
        "regressor__C": np.logspace(-3, 0, 4),
        "regressor__solver": ["lbfgs"],
        "regressor__penalty": ["l2"] # these are actually just the defaults
    }
]

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5)

grid = GridSearchCV(
    pipe,
    param_grid,
    scoring="recall",
    cv=cv,
    return_train_score=True,
    verbose=10
)


### Scoring ###
def print_model_scores(model,
                       scoring_list = [
                           accuracy_score,
                           precision_score,
                           recall_score,
                           f1_score,
                           fbeta_score,
                           roc_auc_score
                        ], 
                       scoring_names = [
                           "accuracy",
                           "precision",
                           "recall",
                           "f1",
                           "fbeta",
                           "ROC AUC"
                        ],
                       X_train = X_train,
                       X_test = X_test,
                       y_train = y_train,
                       y_test = y_test):
    """
    Description
    -----------
    Calculates and prints out scores: recall, f-beta and roc-auc (Later when refactoring
    could parameterise scoring methods too)
    
    Args
    ----
    model: must be a sklearn model object with a predict function
    scoring: list of scoring metrics (MUST BE IMPORTED OR WILL RAISE ERROR)
    """
    # check same number of scoring metrics and names
    assert len(scoring_list) == len(scoring_names), "number of metric names does not equal number of metrics!"

    # prediction
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    # confusion matrix
    print(
        f"""
        Training confusion matrix 
        {confusion_matrix(y_train, train_predictions)}
        Test confusion matrix
        {confusion_matrix(y_test, test_predictions)}
        """
    )

    # iterate through scoring metrics and print respective results
    for name, metric in zip(scoring_names, scoring_list):
        if name == "fbeta": # treat f-beta separately as it has another argument
            train_score = metric(y_train, train_predictions, beta=BETA)
            test_score = metric(y_test, test_predictions, beta=BETA)
        else:
            train_score = metric(y_train, train_predictions)
            test_score = metric(y_test, test_predictions)
        
        print(f"{name}  | Train: {train_score}  | Test: {test_score}")
    

if __name__=="__main__":

    # fit the model
    grid.fit(X_train, y_train)

    print(grid.best_estimator_["regressor"].get_params())

    # transform test data
    X_test_scaled = grid.best_estimator_["scaler"].transform(X_test)

    # score the model
    print_model_scores(grid, X_test=X_test_scaled)

    # feature importance    
    coef_idx_sorted = grid.best_estimator_["regressor"].coef_[0].argsort()
    coef_list = grid.best_estimator_["regressor"].coef_[0]

    f, ax = plt.subplots(1, 1, figsize=(10,7))

    plt.barh(X.columns[coef_idx_sorted], coef_list[coef_idx_sorted])

    f.savefig("Logistic regression feature importance")
