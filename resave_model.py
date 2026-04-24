
import pandas as pd

import numpy as np

import joblib

import sklearn

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

print('sklearn:', sklearn.__version__)

df = pd.read_csv('pipeline_data_realistic.csv')

df['pressure_diff']      = df['pressure_in'] - df['pressure_out']

df['flow_diff']          = df['flow_in'] - df['flow_out']

df['pressure_roll_mean'] = df['pressure_diff'].rolling(5).mean()

df['flow_roll_std']      = df['flow_diff'].rolling(5).std()

df = df.dropna()

features = ['pressure_diff','flow_diff','pressure_roll_mean','flow_roll_std']

X, y = df[features], df['leak']

X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.2, stratify=y, random_state=42

)

model = RandomForestClassifier(class_weight='balanced', random_state=42)

model.fit(X_train, y_train)

joblib.dump(model, 'model.pkl')

print('model.pkl resaved successfully')

