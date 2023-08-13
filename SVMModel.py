import numpy as np
import pandas as pd

dataset = pd.read_table('/kaggle/input/ml-course-ku-3927220000-p003/P003.MArumugam.Data.DE-CRC_species.tsv')
dataset_US = pd.read_table('/kaggle/input/ml-course-ku-3927220000-p003/P003.MArumugam.Data.US-CRC_species.tsv')
dataset_AT = pd.read_table('/kaggle/input/ml-course-ku-3927220000-p003/P003.MArumugam.Data.AT-CRC_species.tsv')
dataset_FR = pd.read_table('/kaggle/input/ml-course-ku-3927220000-p003/P003.MArumugam.Data.FR-CRC_species.tsv')
dataset_CN = pd.read_table('/kaggle/input/ml-course-ku-3927220000-p003/P003.MArumugam.Data.CN-CRC_species.tsv')

X = dataset.iloc[:, 2:].values
y = dataset.iloc[:, 0].values
y = np.where(y == 'control', 0, 1)

count_ones = y.tolist().count(1)
percentage_ones = (count_ones / len(y)) * 100
print("Number of occurrences of 1:", count_ones)
print("Percentage of CRC patients in dataset:", percentage_ones)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 48)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

parameters_dictionary = {'kernel':['linear', 'rbf', 'poly', 'sigmoid'], 
                         'C':[0.0001, 1, 10], 
                         'gamma':[1, 10, 100]}
svc = SVC()


grid_search = GridSearchCV(svc, 
                           parameters_dictionary,
                           scoring = 'f1',
                           return_train_score=True, 
                           cv = 5,
                           verbose = 1) # Displays how many combinations of parameters and folds we'll have, for more information as the time to run each search, use 2 or 3 values instead of 1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
print('The best model was:', best_model)
print('The best parameter values were:', best_parameters)
print('The best accuracy-score was:', best_accuracy)

svm_model = SVC(C=1, gamma=1, kernel='sigmoid')
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)
from sklearn.metrics import accuracy_score, f1_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
f1 = f1_score(y_test, y_pred)
print("f1:", f1)

import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance

# Assuming you have trained and fitted a sigmoid SVM model named 'svm_model'
# Assuming you have the training data stored in X_train and the corresponding labels in y_train

# Compute permutation importance
result = permutation_importance(svm_model, X_train, y_train, n_repeats=10, random_state=42)

# Get feature importances
importances = result.importances_mean
feature_names = dataset.columns[1:]

# Sort feature importances in descending order
sorted_indices = importances.argsort()[::-1]
top_feature_indices = sorted_indices[:10]
top_importances = importances[top_feature_indices]
top_feature_names = [feature_names[idx] for idx in top_feature_indices]

# Plot the top 10 most important features
plt.figure(figsize=(10, 6))
plt.barh(range(len(top_feature_names)), top_importances, align='center')
plt.yticks(range(len(top_feature_names)), top_feature_names)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Top 10 Most Important Features')
plt.show()

X_AT = dataset_AT.iloc[:, 2:].values
y_AT = dataset_AT.iloc[:, 0].values
y_AT = np.where(y_AT == 'control', 0, 1)
X_FR = dataset_FR.iloc[:, 2:].values
y_FR = dataset_FR.iloc[:, 0].values
y_FR = np.where(y_FR == 'control', 0, 1)
X_CN = dataset_CN.iloc[:, 2:].values
y_CN = dataset_CN.iloc[:, 0].values
y_CN = np.where(y_CN == 'control', 0, 1)
X_US = dataset_US.iloc[:, 2:].values
y_US = dataset_US.iloc[:, 0].values
y_US = np.where(y_US == 'control', 0, 1)

X_AT = sc.transform(X_AT)
X_US = sc.transform(X_US)
X_CN = sc.transform(X_CN)
X_FR = sc.transform(X_FR)

y_predAT = svm_model.predict(X_AT)
y_predFR = svm_model.predict(X_FR)
y_predCN = svm_model.predict(X_CN)
y_predUS = svm_model.predict(X_US)

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
    [0.5769230769230769, 0.5614035087719298, 0.5688073394495413, 0.6171875, 0.5833333333333334],
    [0.5769230769230769, 0.6206896551724138, 0.6788990825688074, 0.6083333333333333, 0.65625],
    [0.5096153846153846, 0.5, 0.7142857142857143, 0.4666666666666667, 0.6171875],
    [0.5288461538461539, 0.6491228070175439, 0.6055045871559633, 0.7, 0.53125 ],
    [0.5, 0.5175438596491229,0.48623853211009177,0.675, 0.53125]
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
print("Average accuracy of the models using SVM:", average_diagonal)
