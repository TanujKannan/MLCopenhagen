import numpy as np
import pandas as pd

dataset = pd.read_table('/kaggle/input/ml-course-ku-3927220000-p003/P003.MArumugam.Data.DE-CRC_species.tsv')
dataset_US = pd.read_table('/kaggle/input/ml-course-ku-3927220000-p003/P003.MArumugam.Data.US-CRC_species.tsv')
dataset_AT = pd.read_table('/kaggle/input/ml-course-ku-3927220000-p003/P003.MArumugam.Data.AT-CRC_species.tsv')
dataset_FR = pd.read_table('/kaggle/input/ml-course-ku-3927220000-p003/P003.MArumugam.Data.FR-CRC_species.tsv')
dataset_CN = pd.read_table('/kaggle/input/ml-course-ku-3927220000-p003/P003.MArumugam.Data.CN-CRC_species.tsv')

X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values
y = np.where(y == 'control', 0, 1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Create the Random Forest Classifier
rf_classifier = RandomForestClassifier()

# Define the parameter grid
param_grid = {
    'n_estimators': [10, 20, 30, 100, 200],
    'max_depth': [5, 10, 15],
    'random_state' : [42]
}

# Create the GridSearchCV object
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, scoring='accuracy')

# Perform the grid search on your data
grid_search.fit(X_train, y_train)

# Get the best model and its parameters
best_model = grid_search.best_estimator_
best_parameters = grid_search.best_params_
best_score = grid_search.best_score_

# Print the best model and its parameters
print("Best Model:", best_model)
print("Best Parameters:", best_parameters)
print("Best Accuracy:", best_score)

rf_model = RandomForestClassifier(max_depth = 5, n_estimators = 200, random_state = 42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

from sklearn.metrics import accuracy_score, f1_score
accuracy = accuracy_score(y_test, y_pred)
print("The accuracy of the model is: ", accuracy)

f1 = f1_score(y_test, y_pred)
print("The f1 score of the model is: ", f1)

import matplotlib.pyplot as plt
importances = rf_model.feature_importances_

# Get the indices of the top 5 feature importances
top_indices = np.argsort(importances)[-10:]
top_importances = importances[top_indices]

# Get the corresponding feature names for the top 5 features
feature_names = dataset.columns[1:]  # Assuming dataset is a pandas DataFrame

top_feature_names = feature_names[top_indices]

# Plot the bar chart
plt.barh(top_feature_names, top_importances)
plt.show()

X_AT = dataset_AT.iloc[:, 1:].values
y_AT = dataset_AT.iloc[:, 0].values
y_AT = np.where(y_AT == 'control', 0, 1)
X_FR = dataset_FR.iloc[:, 1:].values
y_FR = dataset_FR.iloc[:, 0].values
y_FR = np.where(y_FR == 'control', 0, 1)
X_CN = dataset_CN.iloc[:, 1:].values
y_CN = dataset_CN.iloc[:, 0].values
y_CN = np.where(y_CN == 'control', 0, 1)
X_US = dataset_US.iloc[:, 1:].values
y_US = dataset_US.iloc[:, 0].values
y_US = np.where(y_US == 'control', 0, 1)

X_AT = sc.transform(X_AT)
X_US = sc.transform(X_US)
X_CN = sc.transform(X_CN)
X_FR = sc.transform(X_FR)

y_predAT = rf_model.predict(X_AT)
y_predFR = rf_model.predict(X_FR)
y_predCN = rf_model.predict(X_CN)
y_predUS = rf_model.predict(X_US)

from sklearn.metrics import accuracy_score, f1_score
accuracy_AT = accuracy_score(y_AT, y_predAT)
print("Accuracy of DE on AT:", accuracy_AT)

from sklearn.metrics import accuracy_score, f1_score
accuracy_US = accuracy_score(y_US, y_predUS)
print("Accuracy of DE on US:", accuracy_US)

from sklearn.metrics import accuracy_score, f1_score
accuracy_FR = accuracy_score(y_FR, y_predFR)
print("Accuracy of DE on FR:", accuracy_FR)

from sklearn.metrics import accuracy_score, f1_score
accuracy_CN = accuracy_score(y_CN, y_predCN)
print("Accuracy of DE on CN:", accuracy_CN)

import numpy as np

# Accuracy values
accuracy_values = [
    [0.6923076923076923, 0.5964912280701754, 0.6513761467889908, 0.575, 0.5390625],
    [0.6153846153846154, 0.7241379310344828, 0.6788990825688074, 0.675, 0.7109375],
    [0.5769230769230769, 0.7280701754385965, 0.8571428571428571, 0.6583333333333333, 0.6953125],
    [0.5865384615384616, 0.6578947368421053, 0.5504587155963303, 0.8666666666666667, 0.6328125],
    [0.6730769230769231, 0.6052631578947368, 0.6055045871559633, 0.75, 0.71875]
]

# Row and column names
row_names = ['US', 'FR', 'AT', 'DE', 'CN']
column_names = ['US', 'FR', 'AT', 'DE', 'CN']

# Create a 5x5 table
table = np.zeros((5, 5))

# Fill in the table values
for i in range(5):
    for j in range(5):
        table[i, j] = accuracy_values[i][j]

# Calculate row-wise averages using np.mean(axis=1)
averages = np.mean(table, axis=1)

# Add the 'Average' column to the table
table = np.column_stack((table, averages))

# Update the column names to include 'Average'
column_names.append('Average')

# Print the updated table with row and column names
print("Accuracy Table:")
header = '\t'.join(column_names)
print(f"\t{header}")
for i in range(5):
    row_values = '\t'.join([f"{val:.4f}" for val in table[i]])
    print(f"{row_names[i]}\t{row_values}")


diagonal_values = np.diagonal(table)
average_diagonal = np.mean(diagonal_values)
# Print the average of the diagonal values
print("Average accuracy of the models using Random Forest:", average_diagonal)
