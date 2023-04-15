# from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.impute import SimpleImputer

housing = pd.read_csv('data.csv')

# train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
# print(f"Rows of the train set: {len(train_set)}\nRows of the test set: {len(test_set)}")

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index] 
    
housing = strat_train_set.drop("MEDV", axis=1)
housing_labels = strat_train_set["MEDV"].copy()

my_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler())
])


# model = LinearRegression()
# model = DecisionTreeRegressor()
model = RandomForestRegressor()


X_test_prepared = my_pipeline.transform()