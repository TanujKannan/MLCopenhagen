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

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

# Define the parameter grid
parameters = {
    'n_neighbors': [3, 5, 7,9,11,13,15,17,19,21],
    'weights': ['uniform', 'distance'],  # Weight function
    'metric': ['euclidean', 'manhattan'],  # Number of neighbors
}

# Create the KNN classifier
knn = KNeighborsClassifier()

# Create the GridSearchCV object
grid_search = GridSearchCV(knn, parameters, scoring='accuracy', cv=5)

# Fit the data to perform the grid search
grid_search.fit(X_train, y_train)

# Retrieve the best model and best parameter values
best_model = grid_search.best_estimator_
best_parameters = grid_search.best_params_
best_score = grid_search.best_score_

# Print the best model and best parameters
print("Best Model:")
print(best_model)
print("Best Parameters:")
print(best_parameters)
print("The best score is:", best_score)

knn_model = KNeighborsClassifier(n_neighbors = 9, metric = 'manhattan', weights = 'uniform')
knn_model.fit(X_train, y_train)

y_pred = knn_model.predict(X_test)

from sklearn.metrics import accuracy_score, f1_score
accuracy = accuracy_score(y_test, y_pred)
print("The accuracy of DE on DE is:", accuracy)

f1 = f1_score(y_test, y_pred)
print("The f1 score is:", f1)

import matplotlib.pyplot as plt

# Calculate baseline accuracy
baseline_accuracy = knn_model.score(X_test, y_test)

# Calculate feature importances using permutation importance
importances = []
for feature in range(X.shape[1]):
    X_permuted = X_test.copy()
    np.random.shuffle(X_permuted[:, feature])
    permuted_accuracy = knn_model.score(X_permuted, y_test)
    importance = baseline_accuracy - permuted_accuracy
    importances.append(importance)

# Get the indices of the top 10 feature importances
top_indices = np.argsort(importances)[-10:]
top_importances = np.array(importances)[top_indices]

# Get the corresponding feature names for the top 10 features
feature_names = dataset.columns[1:]  # Assuming dataset is a pandas DataFrame
top_feature_names = np.array(feature_names)[top_indices]

# Plot the bar chart
plt.figure(figsize=(10, 6))
plt.barh(top_feature_names, top_importances)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Top 10 Feature Importances')
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

y_predAT = knn_model.predict(X_AT)
y_predFR = knn_model.predict(X_FR)
y_predCN = knn_model.predict(X_CN)
y_predUS = knn_model.predict(X_US)

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

# Accuracy values
accuracy_values = [
    [0.6153846153846154, 0.5526315789473685, 0.6330275229357798, 0.5390625, 0.5916666666666667],
    [0.5192307692307693, 0.5517241379310345, 0.6055045871559633, 0.4921875,0.55],
    [0.5096153846153846, 0.6491228070175439, 0.6071428571428571,0.515625, 0.5583333333333333],
    [0.5576923076923077,0.5526315789473685, 0.5871559633027523, 0.40625, 0.5166666666666667],
    [0.5673076923076923, 0.6842105263157895, 0.5963302752293578, 0.6875, 0.8]
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
print("Average accuracy of the models using KNN:", average_diagonal)
