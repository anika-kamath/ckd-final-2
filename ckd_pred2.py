import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score , train_test_split
from sklearn.metrics import classification_report
Df = pd.read_csv('kidney_dataset.csv')
dff= pd.read_csv('kidney_dataset.csv')

X = Df.loc[:, Df.columns != 'classification']

y = Df['classification']

dff = pd.DataFrame(X)
min_max = MinMaxScaler()
min_max.fit(X)
X = min_max.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
from sklearn.ensemble import RandomForestClassifier
rdc = RandomForestClassifier()
rdc.fit(X_train, y_train)
#rdc.score(X_test, y_test)
import pickle
import warnings
warnings.filterwarnings('ignore')
import math


pickle.dump(rdc, open('model2.pkl', 'wb'))
model = pickle.load(open('model2.pkl', 'rb'))
print(model.score(X_test, y_test))
