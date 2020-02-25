import pandas as pd
import numpy as np
import random
import xgboost
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

regressor = DecisionTreeRegressor(criterion='mse', splitter='best')

"""
    Impute na values of dataframe.
    df: Input dataframe to impute.
    columns: List of columns to apply imputation to. Given as list of strings.
    data_types: Data types of given columns. Given as list of strings. Use
        'numerical' for numerical data and 'categorical' for categorical data.
    model: Model used for imputation. Given as a string. Use 'k_nearest_neighbors'
        for k nearest neighbors, 'decision_tree' for decision tree, and 
        'random_forest' for random forest.
    model_params: Parameters to use when created estimation models. Given as a 
        set containing parameter key-value pairs for selected sklearn model.
    
    Returns transformed dataframe.
"""
def impute_nan(df, columns='auto', data_types='auto', model='xgboost', model_params=None):
    if columns is 'auto' and data_types is 'auto':
        # find catagorical and numerical columns
        numerical_columns = df._get_numeric_data().columns
        categorical_columns = list(set(df.columns) - set(numerical_columns))

        # replace NaN values with strings
        df = df.replace(np.nan, 'NaN', regex=True)

        # create categorical encoders
        label_encoders = []
        for i in range(len(categorical_columns)):
            label_encoders.append(LabelEncoder())
            label_encoders[i].fit(df[categorical_columns[i]].astype(str))

        # encode categorical columns
        for i in range(len(categorical_columns)):
            df[categorical_columns[i]] = label_encoders[i].transform(df[categorical_columns[i]])

    if model is 'xgboost':
        regressors = []
        classifiers = []

        # iterate through numerical columns
        for numerical_column in numerical_columns:
            # create XGBoost regressor for each numerical column
            xgboost_regressor = xgboost.XGBRegressor(objective ='reg:squarederror')

            # replace NaN strings with np.nan
            df = df.replace('NaN', np.nan)
            
            # find rows not null for each numerical column
            rows = np.where(~df[numerical_column].isnull())

            # divide dataframe into x and y
            x = df.loc[rows[0], df.columns != numerical_column].to_numpy()
            y = df.loc[rows[0], numerical_column].to_numpy()

            # fit regressor on x and y and append to regressors
            xgboost_regressor.fit(x, y)
            regressors.append(xgboost_regressor)

        # iterate through categorical columns
        for categorical_column in categorical_columns:
            # create XGBoost classifier for each categorical column
            xgboost_classifier = xgboost.XGBClassifier(objective='binary:logistic')

            # transform values of categorical column using label encoder
            transformed_values = label_encoders[i].inverse_transform(df[categorical_column])
            
            # find rows of null values for each categorical column
            rows = [i for i, x in enumerate(transformed_values) if x != 'NaN']

            # divide dataframe into x and y
            x = df.loc[rows, df.columns != categorical_column].to_numpy()
            y = df.loc[rows, categorical_column].to_numpy()

            # fit regressor on x and y and append to regressors
            xgboost_classifier.fit(x, y)
            regressors.append(xgboost_classifier)

            classifiers.append(xgboost_classifier)

        # iterate through numerical columns
        for i in range(len(numerical_columns)):
            # find rows of null values for each numerical column
            rows = np.where(df[numerical_columns[i]].isnull())
            
            # divide dataframe into x
            x = df.loc[rows[0], df.columns != numerical_columns[i]].to_numpy()

            # predict y value based on trained regressor
            y = regressors[i].predict(x)
            
            # replace dataframe NaN values with predicted y values
            for row in rows:
                df.loc[row, numerical_columns[i]] = y[i]

        # iterate through categorical columns
        for i in range(len(categorical_columns)):
            # transform values of categorical column using label encoder
            transformed_values = label_encoders[i].inverse_transform(df[categorical_columns[i]])
            
            # find rows of null values for each categorical column
            rows = [i for i, x in enumerate(transformed_values) if x == 'NaN']
            
            # divide dataframe into x
            x = df.loc[rows, df.columns != categorical_columns[i]].to_numpy()

            # predict y value based on trained classifier
            y = classifiers[i].predict(x)
            y = label_encoders[i].inverse_transform(y)
            
            # replace dataframe NaN values with predicted y values
            for row in rows:
                df.loc[row, categorical_columns[i]] = y[i]

            # find rows containing non null values for each categorical column
            rows = [i for i, x in enumerate(transformed_values) if x != 'NaN']

            # transform encoded labels back to original
            for row in rows:
                df.loc[row, categorical_columns[i]] = label_encoders[i].classes_[df.loc[row, categorical_column]]

        return df
    
    elif model is 'decision_tree':
        print('hello')
    
    elif model is 'k_nearest_neighbors':
        print('hello 3333')

    else:
        raise Exception('Input "{}" for model not recognized.'.format(model))



# load csv
iris = pd.read_csv('iris.csv', index_col=False)

# make copy of iris
iris_na = iris

# remove random 20% of values
ix = [(row, col) for row in range(iris_na.shape[0]) for col in range(iris_na.shape[1])]
for row, col in random.sample(ix, int(round(.2*len(ix)))):
    iris_na.iat[row, col] = np.nan

iris = pd.read_csv('iris.csv', index_col=False)

iris_na = impute_nan(iris_na)

with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(iris)

with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(iris_na)