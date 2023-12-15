#%%
# Predict the review rating a customer is likely to give 
# based on their Age, Gender, Item Purchased, Purchase Amount, 
# Location, Previous Purchases, Frequency of Purchases. 
# The goal of predicting the review rating is to understand the 
# relationships between these features and the numerical review 
# rating provided by customers.

#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score

# Load the dataset
file_path = './shopping_behavior_updated.csv'
data = pd.read_csv(file_path)

# Copying the dataset to avoid modifying the original data
preprocessed_data = data.copy()

# Check missing data
preprocessed_data.isnull().sum()

# Plot the review ratings to visualize the distribution
plt.hist(preprocessed_data["Review Rating"], bins= 20, edgecolor='black', alpha=0.7)
plt.xticks(np.arange(1.0, 6.0))
plt.title("Distribution Of The Review Rating")
plt.xlabel("Review Ratings")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# Encoding categorical variables using LabelEncoder
label_encoders = {}
categorical_columns = ['Gender', 'Item Purchased', 'Category', 'Location', 'Size', 'Color', 
                       'Season', 'Subscription Status', 'Shipping Type', 'Discount Applied', 
                       'Promo Code Used', 'Payment Method', 'Frequency of Purchases']

for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    preprocessed_data[column] = label_encoders[column].fit_transform(preprocessed_data[column])

# Normalizing numerical variables
numerical_columns = ['Age', 'Purchase Amount (USD)', 'Previous Purchases', "Review Rating"]
scaler = StandardScaler()
preprocessed_data[numerical_columns] = scaler.fit_transform(preprocessed_data[numerical_columns])

# Displaying the first few rows of the preprocessed dataset
print(preprocessed_data.head())

#%%
# Define the feature columns and target column
feature_cols = ["Age", "Gender", "Item Purchased",
                "Purchase Amount (USD)", "Location",
                "Previous Purchases", "Frequency of Purchases",
                'Discount Applied', 'Promo Code Used']
target_col = "Review Rating"

X = preprocessed_data[feature_cols]
y = preprocessed_data[target_col]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

#%% 
# Create a linear regression model
lr_model = LinearRegression()

# Define hyperparameters to tune
lr_param_grid = {'fit_intercept': [True, False], 'normalize': [True, False]}

# Use k-fold cross-validation for hyperparameter tuning
lr_kf = KFold(n_splits=9, shuffle=True, random_state=40)

# GridSearchCV for hyperparameter tuning
lr_grid_search = GridSearchCV(lr_model, lr_param_grid, scoring='neg_mean_squared_error', cv=lr_kf)
lr_grid_search.fit(X_train, y_train)

# Display the best hyperparameters and best model
print("Best Hyperparameters:", lr_grid_search.best_params_)
best_lr_model = lr_grid_search.best_estimator_

# Evaluate the best model on the test set
lr_y_pred = best_lr_model.predict(X_test)

lr_mse = mean_squared_error(y_test, lr_y_pred)
lr_mae = mean_absolute_error(y_test, lr_y_pred)
lr_r2 = r2_score(y_test, lr_y_pred)
lr_explained_variance = explained_variance_score(y_test,lr_y_pred)

# Display the score results
print(f"{lr_model.__class__.__name__}: ")
print(f"Mean Squared Error is {lr_mse:.2f}")
print(f"Mean Absolute Error is {lr_mae:.2f} ")
print(f"R-squared is {lr_r2:.2f}")
print(f"Explained Variance Score is {lr_explained_variance: .2f} \n")

# Get coefficients of the linear regression model as 
# an indicator of feature importance.
coefficients = lr_model.fit(X, y).coef_

# Print feature coefficients
print("Feature coefficients:")
for feature, coefficient in zip(X_train.columns, coefficients):
    print(f"{feature}: {coefficient}")

#%%
# Define the RandomForestRegressor
rf_model = RandomForestRegressor()

# Define the parameter grid to search
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create a KFold cross-validation object
rf_kf = KFold(n_splits=3, shuffle=True, random_state=15)

# Create the GridSearchCV object
rf_grid_search = GridSearchCV(estimator=rf_model, param_grid=rf_param_grid, scoring='neg_mean_squared_error', cv=rf_kf)

# Fit the model with the data
rf_grid_search.fit(X, y)

# Get the best parameters and best model
rf_best_params = rf_grid_search.best_params_
rf_best_model = rf_grid_search.best_estimator_

# Print the best parameters
print(f"Best Parameters: {rf_best_params}")

# Use the best model to make predictions
rf_y_pred = rf_best_model.predict(X_test)

# Evaluate the model
rf_mse = mean_squared_error(y_test, rf_y_pred)
rf_mae = mean_absolute_error(y_test, rf_y_pred)
rf_r2 = r2_score(y_test, rf_y_pred)
rf_explained_variance = explained_variance_score(y_test, rf_y_pred)

# Display the score results
print(f"{rf_model.__class__.__name__}:")
print(f"Mean Squared Error is {rf_mse:.2f}")
print(f"Mean Absolute Error is {rf_mae:.2f} ")
print(f"R-squared is {rf_r2:.2f}")
print(f"Explained Variance Score is {rf_explained_variance:.2f} /n")

# Get feature importances
importances = rf_model.fit(X, y).feature_importances_
# Sort indices in descending order of importance
indices = np.argsort(importances)[::-1]
# Print feature ranking
print("Feature ranking:")
for f in range(X_train.shape[1]):
    print(f"{X_train.columns[indices[f]]}: {importances[indices[f]]}")
print("------------------------------------")

#%% Visualize the comparison scores of two models
# Create lists for each metric and model
metrics = ['Mean Squared Error (MSE)', 'Mean Absolute Error (MAE)', 'R-squared', 'Explained Variance']
lr_scores = [lr_mse, lr_mae, lr_r2, lr_explained_variance]
rf_scores = [rf_mse, rf_mae, rf_r2, rf_explained_variance]

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))

bar_width = 0.35
index = range(len(metrics))

bar1 = ax.bar(index, lr_scores, bar_width, label='Linear Regression')
bar2 = ax.bar([i + bar_width for i in index], rf_scores, bar_width, label='Random Forest Regressor')

# Add labels, title, and legend
ax.set_xlabel('Metrics')
ax.set_ylabel('Scores')
ax.set_title('Comparison of Model Scores')
ax.set_xticks([i + bar_width / 2 for i in index])
ax.set_xticklabels(metrics)
ax.legend()

# Display the plot
plt.show()

#%%
# Results:
# LinearRegression: 
# Mean Squared Error is 0.98
# Mean Absolute Error is 0.85 
# R-squared is -0.01
# Explained Variance Score is -0.01

# RandomForestRegressor:
#RandomForestRegressor:
#Mean Squared Error is 0.66
#Mean Absolute Error is 0.69 
#R-squared is 0.32
#Explained Variance Score is 0.32
#
# In summary, based on the provided scores, the Random Forest Regressor 
# outperforms the Linear Regression model in terms of predictive performance, 
# as indicated by lower MSE, higher R-squared, and Explained Variance Score.
#%%
