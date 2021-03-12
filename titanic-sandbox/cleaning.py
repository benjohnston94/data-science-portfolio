import pandas as pd
import pickle

# Types
from pandas import DataFrame # for type hints
from typing import List


# def drop_rows(df: DataFrame,
#                cols: List[str] = ["Embarked"]):
#     """
#     For specific columns with only a couple of null values, 
#     drop these values for ease
#     """
#     for col in cols:
#         df = df.loc[raw[col].notnull()].copy()
    
#     return df

def impute_incomplete_cols(df: DataFrame,
                           cols: List[str] = ["Age"],
                           impute_method: str = "median") -> DataFrame:
    """
    for a given set of columns, imput the value of the specified type
    """
    for col in cols:
        if impute_method == "median":
            df[col].fillna(df[col].median(), inplace=True)
        elif impute_method == "mean":
            df[col].fillna(df[col].mean(), inplace=True)
        elif impute_method == "min":
            df[col].fillna(df[col].min(), inplace=True)
        elif impute_method == "max":
            df[col].fillna(df[col].max(), inplace=True)
        else:
            print(f"Unknown impute method: {impute_method}")
            
    return df

def missing_to_boolean(df: DataFrame,
                       cols: List[str] = ["Cabin"]) -> DataFrame:
    """
    for columns with too many missing values we just assign
    a value to specify if null or not
    """
    for col in cols:
        df[f"{col}_value_present"] = df[col].notnull()
    
    return df

def convert_to_dummy(df: DataFrame,
                     cols: List[str] = ["Sex", "Embarked"]) -> DataFrame:
    """
    Convert specified categorical columns to dummy variables
    """
    for col in cols:
        dummies = pd.get_dummies(df[col], drop_first=True)
        df = df.join(dummies).copy()
    
    return df



def drop_cols(df: DataFrame,
              cols: List[str] = ["PassengerId", "Name", "Sex", "Ticket", "Cabin", "Embarked"]) -> DataFrame:
    """
    Put this at the end so can run pipeline without 
    functions but still remove unusable columns. 
    Includes all columns converted to dummy variables
    """
    df.drop(cols, axis=1, inplace=True)
    
    return df

def drop_rows(df: DataFrame) -> DataFrame:
    """
    Drop all rows with nulls remaining.
    Putting this at the end catches anything which had nulls to begin with and 
    will also drop any related columns that manipulated these rows (e.g. dummy variables)
    """
    df.dropna()
    
    return df

def read_and_clean_data(file_path: str = "titanic_train.csv") -> DataFrame:
    """
    Reads in titanic file and cleans the data
    """
    raw = pd.read_csv(file_path)

    clean = (
        raw             
        .pipe(impute_incomplete_cols)
        .pipe(missing_to_boolean)
        .pipe(convert_to_dummy)
        .pipe(drop_cols)
        .pipe(drop_rows)
    )

    assert clean.isna().sum().sum() == 0, "still null values"
    
    return clean

def pickle_clean_data(df: DataFrame,
                      save_target: str = "titanic_train_clean"):
    """
    Takes in a dataframe and pickles it
    with the given name
    """
    df.to_pickle(save_target)
    
    return


if __name__=="__main__":
    df = read_and_clean_data()
    pickle_clean_data(df)
