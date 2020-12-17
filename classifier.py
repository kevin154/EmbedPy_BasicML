
# https://machinelearningmastery.com/develop-first-xgboost-model-python-scikit-learn/
# https://www.kaggle.com/uciml/pima-indians-diabetes-database

import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
dataset = np.loadtxt('C:/q/w64/pima-natives-diabetes.csv', delimiter=",")

# Split data into train and test sets, leave 33% of entries aside for testing
X = dataset[:,0:8]
y = dataset[:,8]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# Fit model with training data
model = XGBClassifier()
model.fit(X_train, y_train)

# Make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

# Evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: {:.2f}%".format(accuracy * 100.0))
