# Customer/Loan Approval Prediction Pipeline


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
)


df = pd.read_csv("loan_approval.csv")

print("Dataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())


df['credit_rating'] = pd.cut(
    df['credit_score'],
    bins=[0, 580, 670, 740, 850],
    labels=['Poor', 'Fair', 'Good', 'Excellent']
)



df.loc[
    df.sample(frac=0.05, random_state=42).index,
    'income'
] = np.nan

df.loc[
    df.sample(frac=0.05, random_state=1).index,
    'city'
] = np.nan

df.loc[
    df.sample(frac=0.05, random_state=7).index,
    'credit_rating'
] = np.nan


X = df.drop(['loan_approved'], axis=1)

y = df['loan_approved']



ordinal_features = ['credit_rating']

nominal_features = ['city']

numerical_features = [
    'income',
    'credit_score',
    'loan_amount',
    'years_employed',
    'points'
]



numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean'))
])



ordinal_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(
        categories=[
            ['Poor', 'Fair', 'Good', 'Excellent']
        ]
    ))
])


nominal_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])



preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numerical_features),
    ('ord', ordinal_transformer, ordinal_features),
    ('nom', nominal_transformer, nominal_features)
])



model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(
        max_iter=5000,
        random_state=42
    ))
])



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)



model_pipeline.fit(X_train, y_train)



y_pred = model_pipeline.predict(X_test)



print("\nAccuracy Score:")
print(round(accuracy_score(y_test, y_pred), 4))