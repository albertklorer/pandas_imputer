import pandas as pd

"""
    Impute na values of dataframe.
    df: Input dataframe to impute.
    columns: List of columns to apply imputation to. Given as list of strings.
    data_types: Data types of given columns. Given as list of strings. Use
        'numerical' for numerical data and 'categorical' for categorical data.
    method: Method used for imputation. Given as a string. Use 'k_nearest_neighbors'
        for k nearest neighbors and 'decision_tree' for decision tree, and 
        'random_forest' for random forest.
"""
def impute_na(df, columns, data_types, method='decision_tree') -> pd.DataFrame:
    print(df)

iris = pd.read_csv('iris.csv', index_col=False)
print(iris.head)
