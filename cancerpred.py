import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

data = pd.read_csv("data.csv")
data.drop(["Unnamed: 32","id"],axis=1, inplace = True)
data.diagnosis = [1 if value == "M" else 0 for value in data.diagnosis]
"""print(data.head())"""
y = data["diagnosis"]
x = data.drop(["diagnosis"],axis = 1)
scaler = StandardScaler()
xscaled = scaler.fit_transform(x)
print(xscaled)
xtrain , xtest , ytrain , ytest = train_test_split(xscaled, y, test_size= 0.30, random_state=42)
lr = LogisticRegression()
lr.fit(xtrain,ytrain)
ypred = lr.predict(xtest)
accuracy = accuracy_score(ytest,ypred)
print(f"Accuracy: {accuracy: .2f}")
print(classification_report(ytest,ypred))
"""
our model is ready to predict if a tumor is M(malignant) or B(benign)
"""
