# %%
#   Predict the purchase amount by age, gender, location, size, season, item purchased, and color.
#   What is the expected purchase amount for a customer's next transaction based on their previous purchasing behavior,
#   including the frequency of purchases, average review rating,and the types of items purchased?
# %%

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# preprocessing the data
file_path = './shopping_behavior_updated.csv'
data = pd.read_csv(file_path)

# Copying the dataset to avoid modifying the original data
preprocessed_data = data.copy()

# Encoding categorical variables using LabelEncoder
label_encoders = {}
categorical_columns = ['Gender', 'Item Purchased', 'Category', 'Location', 'Size', 'Color',
                       'Season', 'Subscription Status', 'Shipping Type', 'Discount Applied',
                       'Promo Code Used', 'Payment Method', 'Frequency of Purchases']

for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    preprocessed_data[column] = label_encoders[column].fit_transform(
        preprocessed_data[column])

# Normalizing numerical variables
numerical_columns = [
    'Age', 'Purchase Amount (USD)', 'Review Rating', 'Previous Purchases']
scaler = StandardScaler()
preprocessed_data[numerical_columns] = scaler.fit_transform(
    preprocessed_data[numerical_columns])


# %%
# Linear Regression
# Import necessary libraries

# Separate features (X) and target variable (y)
X = preprocessed_data[['Age', 'Gender', 'Location', 'Size', 'Season', 'Item Purchased',
                       'Color', 'Frequency of Purchases', 'Review Rating', 'Previous Purchases']]
y = preprocessed_data['Purchase Amount (USD)']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# # Example: Predicting the purchase amount for a new customer
# new_customer_data = pd.DataFrame({
#     'Age': [30],
#     'Gender': ['Male'],
#     'Location': ['New York'],
#     'Size': ['M'],
#     'Season': ['Fall'],
#     'Item Purchased': ['Sweater'],
#     'Color': ['Blue'],
#     'Frequency of Purchases': [10],
#     'Review Rating': [4.0],
#     'Previous Purchases': [20]
# })

# # Encode categorical variables for the new customer
# for column in categorical_columns:
#     new_customer_data[column] = label_encoders[column].transform(
#         new_customer_data[column])

# # Make a prediction for the new customer
# new_customer_prediction = model.predict(new_customer_data)

# print(
#     f'Predicted Purchase Amount for the New Customer: {new_customer_prediction[0]}')

# %%
# Random Forest Regressor
# Import necessary libraries

# Separate features (X) and target variable (y)
X = preprocessed_data[['Age', 'Gender', 'Location', 'Size', 'Season', 'Item Purchased',
                       'Color', 'Frequency of Purchases', 'Review Rating', 'Previous Purchases']]
y = preprocessed_data['Purchase Amount (USD)']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# # Example: Predicting the purchase amount for a new customer
# new_customer_data = pd.DataFrame({
#     'Age': [30],
#     'Gender': ['Male'],
#     'Location': ['New York'],
#     'Size': ['M'],
#     'Season': ['Fall'],
#     'Item Purchased': ['Sweater'],
#     'Color': ['Blue'],
#     'Frequency of Purchases': [10],
#     'Review Rating': [4.0],
#     'Previous Purchases': [20]
# })

# # Encode categorical variables for the new customer
# for column in categorical_columns:
#     new_customer_data[column] = label_encoders[column].transform(
#         new_customer_data[column])

# # Make a prediction for the new customer
# new_customer_prediction = model.predict(new_customer_data)

# print(
#     f'Predicted Purchase Amount for the New Customer: {new_customer_prediction[0]}')

# %%

# we first used 'Age', 'Gender', 'Location', 'Size', 'Season', 'Item Purchased', 'Color', 'Frequency of Purchases', 'Review Rating', 'Previous Purchases'
# these features to predict the purchase amount but the MSE and R-squared are not good enough.

# we need to find the most important features to predict the purchase amount
