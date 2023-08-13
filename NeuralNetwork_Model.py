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

y_train = y_train.flatten()

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

param_grid = {
    'hidden_layer_sizes': [(50,100,1), (848,400,25,1)],
    'activation': ['relu'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'invscaling', 'adaptive'],
    
}

# Create the neural network classifier
nn_classifier = MLPClassifier(random_state=42, max_iter=3000)

# Initialize the GridSearchCV
grid_search = GridSearchCV(nn_classifier, param_grid, cv=5, scoring='accuracy', verbose=1)

# Fit the GridSearchCV to find the best hyperparameters
grid_search.fit(X_train, y_train)

# Get the best hyperparameters and best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Print the best hyperparameters and accuracy
print("Best Hyperparameters:", best_params)
print("Best Accuracy:", grid_search.best_score_)

# Test the best model on the test set
test_accuracy = best_model.score(X_test, y_test)
print("Test Accuracy:", test_accuracy)

nn_model = MLPClassifier(activation = 'relu', alpha = 0.001, hidden_layer_sizes = (50, 100, 1), learning_rate = 'constant', random_state = 42, max_iter = 3000)
nn_model.fit(X_train,y_train)

import matplotlib.pyplot as plt

# Calculate baseline accuracy
baseline_accuracy = nn_model.score(X_test, y_test)

# Calculate feature importances using permutation importance
importances = []
for feature in range(X.shape[1]):
    X_permuted = X_test.copy()
    np.random.shuffle(X_permuted[:, feature])
    permuted_accuracy = nn_model.score(X_permuted, y_test)
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

y_predAT = nn_model.predict(X_AT)
y_predFR = nn_model.predict(X_FR)
y_predCN = nn_model.predict(X_CN)
y_predUS = nn_model.predict(X_US)

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
    [0.5384615384615384, 0.5263157894736842, 0.6055045871559633,0.6015625, 0.6],
    [0.5576923076923077, 0.6896551724137931, 0.5321100917431193, 0.703125,0.6166666666666667],
    [0.5769230769230769, 0.5087719298245614, 0.75,0.609375, 0.5583333333333333],
    [0.5769230769230769,0.6403508771929824,0.5596330275229358 ,0.78125 ,0.5916666666666667 ],
    [0.5192307692307693, 0.6140350877192983,0.5045871559633027,0.4296875 ,0.7333333333333333 ]
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
print("Average accuracy of the models using NN:", average_diagonal)
