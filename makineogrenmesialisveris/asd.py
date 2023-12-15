# Import the necessary libraries.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error , r2_score
from sklearn.feature_selection import SelectKBest, f_regression

# Load the dataset
data = pd.read_csv('Mall_Customers.csv')

# Examine the dataset
print(data.head())

# Obtain the statistical summary of the dataset
print(data.describe())

# Check for missing values in the dataset
print(data.isnull().sum())

# Set the ID column as the index.
data.set_index('CustomerID', inplace=True)

# Convert the categorical variable "Gender" into a binary feature
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})

# Visualize the dataset
columns = ['Gender', 'Age', 'Annual Income (k$)']
target = 'Spending Score (1-100)'
data = data[columns + [target]]

for column in columns:
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=data, x=column, y=target)
    plt.title(f'{column} vs. {target}')
    plt.show()

    # The correlation matrix between the independent variables and the target variable in the dataset.
correlation_matrix = data.corr()

# Visualize the correlation matrix.
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Korelasyon Matrisi')
plt.show()

# Separate the independent variables and the target variable
X = data.drop(['Spending Score (1-100)'], axis=1)
y = data['Spending Score (1-100)']

# Evaluate the relationship between the independent variables and the target variable.
selector = SelectKBest(f_regression, k=2)
X_new = selector.fit_transform(X, y)

# Retrieve the indices of the best features.
feature_indices = selector.get_support(indices=True)

# Select the relevant features.
selected_features = X.columns[feature_indices]

# Create a new dataset with the selected features.
X_selected = X[selected_features]

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest regression model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model's performance on the training set
y_train_pred = model.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
print("Root Mean Squared Error (RMSE) on the Train Set:", train_rmse)

# Evaluate the model's performance on the test set
y_test_pred = model.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
print("Root Mean Squared Error (RMSE) on the Test Set:", test_rmse)

# Calculate the R2 score
y_train_pred = model.predict(X_train)
train_r2 = r2_score(y_train, y_train_pred)
print("R² Result:", train_r2)

# Compare the actual values with the predicted values.
plt.figure(figsize=(10, 6))
plt.bar(np.arange(len(y_test)), y_test, width=0.4, align='center', label='Real Values')
plt.bar(np.arange(len(y_test)), y_test_pred, width=0.4, align='edge', label='The Predicted Values')
plt.xlabel('Examples')
plt.ylabel('Values')
plt.title('Random Forest Model - Real Values vs. Predicted Values')
plt.legend()
plt.show()
