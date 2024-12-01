# Import necessary libraries
import tensorflow as tf  # For building and training models (not used in this script but imported for potential use)
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical computations

# Load and clean the dataset
data = pd.read_csv('https://raw.githubusercontent.com/Ranjit-Singh-786/MU/refs/heads/master/Student_Performance.csv').drop_duplicates()  # Load data and remove duplicates

# Importing visualization libraries
import matplotlib.pyplot as plt  # For creating plots
import seaborn as sns  # For enhanced data visualization

# Visualize the distribution of the target variable (Performance Index)
y_train.hist(bins=20)  # Histogram for the 'y_train' variable (may raise error if y_train not yet defined)

# Scatter plot to visualize the relationship between Hours Studied and Performance Index
sns.scatterplot(x='Hours Studied', y='Performance Index', data=data)
plt.title('Hours Studied vs. Performance Index')
plt.xlabel('Hours Studied')
plt.ylabel('Performance Index')
plt.show()

# Bar plot to compare Performance Index based on Extracurricular Activities
sns.barplot(x='Extracurricular Activities', y='Performance Index', data=data)
plt.title('Performance Index by Extracurricular Activities')
plt.xlabel('Extracurricular Activities')
plt.ylabel('Performance Index')
plt.show()

# Pie chart to show the distribution of Hours Studied
data['Hours Studied'].value_counts().plot.pie()
plt.title('Distribution of Hours Studied')
plt.ylabel('')
plt.show()

# Map Extracurricular Activities column from Yes/No to numeric values for modeling
activities_dict = {"Yes": 1, "No": 2}  # Define the mapping
data['Extracurricular Activities'] = data['Extracurricular Activities'].map(activities_dict)  # Apply the mapping

# Define the features (X) and target (y) variables
x = data[["Hours Studied", "Extracurricular Activities", "Sleep Hours", "Sample Question Papers Practiced", "Previous Scores"]]  # Feature columns
y = data[["Performance Index"]]  # Target column

# Import machine learning libraries
from sklearn.model_selection import train_test_split  # For splitting data into train and test sets
from sklearn.linear_model import LinearRegression  # Linear Regression model
from sklearn.tree import DecisionTreeRegressor  # Decision Tree model
from sklearn.ensemble import RandomForestRegressor  # Random Forest model
from sklearn.ensemble import AdaBoostRegressor  # AdaBoost model
from sklearn.neighbors import KNeighborsRegressor  # K-Nearest Neighbors model
import xgboost as xgb  # XGBoost model

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)  # 80% training, 20% testing

# Initialize machine learning models
lnr = LinearRegression()  # Linear Regression
dtr = DecisionTreeRegressor()  # Decision Tree
rdf = RandomForestRegressor()  # Random Forest
ada = AdaBoostRegressor(random_state=0, n_estimators=100)  # AdaBoost with 100 estimators
knn = KNeighborsRegressor(n_neighbors=5)  # K-Nearest Neighbors with 5 neighbors
xgb_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)  # XGBoost model

# Train all models on the training data
lnr.fit(x_train, y_train)  # Train Linear Regression
dtr.fit(x_train, y_train)  # Train Decision Tree
rdf.fit(x_train, y_train)  # Train Random Forest
ada.fit(x_train, y_train)  # Train AdaBoost
knn.fit(x_train, y_train)  # Train KNN
xgb_model.fit(x_train, y_train)  # Train XGBoost

# Confirm training completion
print("Successfully trained all algorithms")

# Print training scores for all models
print("Linear Regression Score (Train):", lnr.score(x_train, y_train))
print("Decision Tree Score (Train):", dtr.score(x_train, y_train))
print("Random Forest Score (Train):", rdf.score(x_train, y_train))
print("ADA Boost (Train):", ada.score(x_train, y_train))
print("KNN (Train):", knn.score(x_train, y_train))
print("XGBoost (Train):", xgb_model.score(x_train, y_train))

# Calculate test scores for all models
a = lnr.score(x_test, y_test)  # Test score for Linear Regression
b = dtr.score(x_test, y_test)  # Test score for Decision Tree
c = rdf.score(x_test, y_test)  # Test score for Random Forest
d = ada.score(x_test, y_test)  # Test score for AdaBoost
e = knn.score(x_test, y_test)  # Test score for KNN
f = xgb_model.score(x_test, y_test)  # Test score for XGBoost

# Print test scores for all models
print("Linear Regression Score (Test):", a)
print("Decision Tree Score (Test):", b)
print("Random Forest Score (Test):", c)
print("ADA Boost (Test):", d)
print("KNN (Test):", e)
print("XGBoost (Test):", f)

# Compare all models and find the best one
results = pd.DataFrame({
    'Algorithm': ['Linear Regression', 'Decision Tree', 'Random Forest', 'AdaBoost', 'KNN', 'XGBoost'],  # Model names
    'Test_Score': [a, b, c, d, e, f]  # Corresponding test scores
})

# Identify the algorithm with the highest test score
best_algorithm = results.loc[results['Test_Score'].idxmax()]  # Get row with the best score

# Print the performance summary and the best algorithm
print("Performance Summary:")
print(results)
print("\nBest Algorithm:")
print(best_algorithm)

# Predict using the best algorithm
best_model_name = best_algorithm['Algorithm']  # Get the name of the best algorithm

# Match the best algorithm name to its model instance
if best_model_name == 'Linear Regression':
    best_model = lnr
elif best_model_name == 'Decision Tree':
    best_model = dtr
elif best_model_name == 'Random Forest':
    best_model = rdf
elif best_model_name == 'AdaBoost':
    best_model = ada
elif best_model_name == 'KNN':
    best_model = knn
elif best_model_name == 'XGBoost':
    best_model = xgb_model
else:
    print("Error: Unknown best model name")
    best_model = None

# If a valid best model is identified, make predictions on the test data
if best_model:
    predictions = best_model.predict(x_test)  # Predict using the best model
    y_test['prediction'] = predictions  # Add predictions to the test data
y_test  # Display the updated test dataset with predictions
