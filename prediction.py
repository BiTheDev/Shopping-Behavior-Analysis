# %%
#   Predict the purchase amount by age, gender, location, size, season, item purchased, and color.
#   What is the expected purchase amount for a customer's next transaction based on their previous purchasing behavior,
#   including the frequency of purchases, average review rating,and the types of items purchased?
# %%

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score
import seaborn as sns
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
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
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

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

# Assuming you have already applied LabelEncoder to your categorical columns

for column in categorical_columns:
    le = label_encoders[column]
    encoded_values = le.transform(le.classes_)

    # Print the mapping of original values to encoded values
    print(f"Mapping for {column}:")
    for original_value, encoded_value in zip(le.classes_, encoded_values):
        print(f"{original_value} -> {encoded_value}")
    print("\n")

preprocessed_data.fillna(0, inplace=True)
# Normalizing numerical variables
numerical_columns = [
    'Age', 'Review Rating', 'Previous Purchases']
scaler = StandardScaler()
preprocessed_data[numerical_columns] = scaler.fit_transform(
    preprocessed_data[numerical_columns])

# Check for NaN values in each column
nan_columns = preprocessed_data.columns[preprocessed_data.isna(
).any()].tolist()

# Check for infinite values in each column
inf_columns = preprocessed_data.columns[(
    preprocessed_data.abs() == np.inf).any()].tolist()

# Combine both lists to get columns with either NaN or infinite values
columns_with_issues = list(set(nan_columns + inf_columns))

# Print columns with NaN or infinite values
print("Columns with NaN or infinite values:", columns_with_issues)


# %%
# Need to visualize purchase amount distribution
# Visualizing the distribution of the target variable
plt.hist(preprocessed_data['Purchase Amount (USD)'])
plt.title('Purchase Amount Distribution')
plt.xlabel('Purchase Amount (USD)')
plt.ylabel('Count')
plt.show()
# stats for purchase amount
preprocessed_data['Purchase Amount (USD)'].describe()

# we can group from 20-40, 40-60, 60-80, 80-100
# I want to categorize the purchase amount into 4 groups
# 1: 20-40, 2: 40-60, 3: 60-80, 4: 80-100
# make a new column called 'Purchase Amount Group'

# %%
# selected_columns = ['Age', 'Gender', 'Location',  'Item Purchased',
#                     'Frequency of Purchases', 'Review Rating', 'Previous Purchases']

# X = preprocessed_data[selected_columns]
# y = preprocessed_data['Purchase Amount Group']  # Target variable

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.3, random_state=42)


# # Step 4: Train the KNN model
# k = 10  # Adjust the value of k as needed
# knn_model = KNeighborsClassifier(n_neighbors=k)
# knn_model.fit(X_train, y_train)

# # Step 5: Make predictions
# y_pred = knn_model.predict(X_test)

# # Step 6: Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy:.2f}")

# # Classification report for more detailed evaluation
# print(classification_report(y_test, y_pred))

# %%

# # Assuming 'Target_Variable_Column_Name' is the column you want to predict
# target_variable = 'Purchase Amount Group'

# # Selecting features for clustering (customize based on your dataset)
# selected_features = ['Age', 'Gender', 'Location', 'Size', 'Season', 'Item Purchased',
#                      'Frequency of Purchases', 'Review Rating', 'Previous Purchases']

# # Extracting the selected features
# X = preprocessed_data[selected_features]

# # Standardizing the data
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Determining the optimal number of clusters using the Elbow method
# wcss = []
# for i in range(1, 11):
#     kmeans = KMeans(n_clusters=i, init='k-means++',
#                     max_iter=300, n_init=10, random_state=0)
#     kmeans.fit(X_scaled)
#     wcss.append(kmeans.inertia_)

# # Plotting the Elbow method
# plt.plot(range(1, 11), wcss)
# plt.title('Elbow Method')
# plt.xlabel('Number of Clusters')
# plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
# plt.show()

# # Based on the Elbow method, choose the optimal number of clusters (let's say k=3)
# optimal_clusters = 10

# # Applying K-Means clustering
# kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++',
#                 max_iter=300, n_init=10, random_state=0)
# preprocessed_data['Cluster'] = kmeans.fit_predict(X_scaled)

# # Visualizing the clusters (customize based on your dataset)
# for cluster in range(optimal_clusters):
#     cluster_data = preprocessed_data[preprocessed_data['Cluster'] == cluster]
#     plt.scatter(
#         cluster_data['Age'], cluster_data['Purchase Amount Group'], label=f'Cluster {cluster + 1}')

# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[
#             :, 1], s=300, c='red', marker='*', label='Centroids')
# plt.title('Customer Segmentation')
# plt.xlabel('Gender')
# plt.ylabel('Purchase Amount Group')
# plt.legend()
# plt.show()

# %%
AMOUNT_THRESHOLD = 50
preprocessed_data['Purchase Amount (USD)'].describe()
preprocessed_data['Purchase Amount Group'] = np.where(
    preprocessed_data['Purchase Amount (USD)'] > AMOUNT_THRESHOLD, 1, 0)

preprocessed_data['Purchase Amount Group'].value_counts()

# %%
df = preprocessed_data.copy()
df_segments = df[["Age", "Gender", "Location",
                  "Review Rating", "Category", "Frequency of Purchases"]]
X = preprocessed_data.drop(
    ['Customer ID', 'Category', 'Purchase Amount (USD)', 'Purchase Amount Group'], axis=1)
# X = df_segments
y = preprocessed_data['Purchase Amount Group']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Classification report for more detailed evaluation

print(classification_report(y_test, y_pred))


# Get feature importances
feature_importances = rf_model.feature_importances_

# Display feature importances
print("Feature Importances:")
for feature, importance in zip(X_train.columns, feature_importances):
    print(f"{feature}: {importance:.4f}")

# Display confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# %%
# Fine tuning this model

# Define the hyperparameters and their possible values
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap': [True, False]
}

# Create the Random Forest model
rf_model = RandomForestClassifier(random_state=42)

# Perform grid search with cross-validation
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best hyperparameter values
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Get the best model
best_rf_model = grid_search.best_estimator_

# Make predictions on the test set using the best model
y_pred_tuned = best_rf_model.predict(X_test)

# Evaluate the tuned model
accuracy_tuned = accuracy_score(y_test, y_pred_tuned)
print(f"Tuned Model Accuracy: {accuracy_tuned:.2f}")

# Classification report for more detailed evaluation
print(classification_report(y_test, y_pred_tuned))

# %%
# Define individual classifiers
rf_classifier = RandomForestClassifier(
    n_estimators=100, random_state=42, class_weight='balanced')
svm_classifier = SVC(probability=True)
gb_classifier = GradientBoostingClassifier()

# Create a Voting Classifier
voting_classifier = VotingClassifier(estimators=[
    ('rf', rf_classifier),
    ('svm', svm_classifier),
    ('gb', gb_classifier)
], voting='soft')  # 'soft' allows for probability voting

# Train the Voting Classifier
voting_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred_voting = voting_classifier.predict(X_test)

# Evaluate the Voting Classifier
accuracy_voting = accuracy_score(y_test, y_pred_voting)
print(f"Voting Classifier Accuracy: {accuracy_voting:.2f}")

# %%
# Preprocess the data
# Can we predict the likelihood of a customer making a purchase in "Clothing" Category based on their demographics (Age, Gender), location?


# add a column that shows if person bought a clothing product
# 1: yes, 0: no
# 1 is the clothing category in the dataset

CLOTHING = 2
preprocessed_data['Clothing'] = np.where(
    preprocessed_data['Category'] == CLOTHING, 1, 0)
preprocessed_data['Clothing'].value_counts()


# %%


# Assuming X contains features and y contains the binary target variable 'Clothing'
# Make sure to exclude the original 'Category' column if it's still present in X
df = preprocessed_data.copy()
df_segments = df[["Age", "Gender", "Season", "Subscription Status", "Category",
                  "Purchase Amount (USD)", "Frequency of Purchases"]]
X = preprocessed_data.drop(['Customer ID', 'Category', 'Clothing'], axis=1)
# X = df_segments
y = preprocessed_data['Clothing']
print(X.columns)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Classification report for more detailed evaluation
print(classification_report(y_test, y_pred))

# %%
# looking at the feature importance

# Get feature importances from the trained Random Forest model
feature_importances = rf_model.feature_importances_

# Create a DataFrame to display feature importances
feature_importance_df = pd.DataFrame(
    {'Feature': X.columns, 'Importance': feature_importances})

# Sort the DataFrame by importance in descending order
feature_importance_df = feature_importance_df.sort_values(
    by='Importance', ascending=False)

# Display feature importances
print(feature_importance_df)

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importances')
plt.show()


# %%
# Fine tuning this model

# Define the hyperparameters and their possible values
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap': [True, False]
}

# Create the Random Forest model
rf_model = RandomForestClassifier(random_state=42)

# Perform grid search with cross-validation
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best hyperparameter values
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Get the best model
best_rf_model = grid_search.best_estimator_

# Make predictions on the test set using the best model
y_pred_tuned = best_rf_model.predict(X_test)

# Evaluate the tuned model
accuracy_tuned = accuracy_score(y_test, y_pred_tuned)
print(f"Tuned Model Accuracy: {accuracy_tuned:.2f}")

# Classification report for more detailed evaluation
print(classification_report(y_test, y_pred_tuned))

# %%

# Define individual classifiers
rf_classifier = RandomForestClassifier(
    n_estimators=100, random_state=42, class_weight='balanced')
svm_classifier = SVC(probability=True)
gb_classifier = GradientBoostingClassifier()

# Create a Voting Classifier
voting_classifier = VotingClassifier(estimators=[
    ('rf', rf_classifier),
    ('svm', svm_classifier),
    ('gb', gb_classifier)
], voting='soft')  # 'soft' allows for probability voting

# Train the Voting Classifier
voting_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred_voting = voting_classifier.predict(X_test)

# Evaluate the Voting Classifier
accuracy_voting = accuracy_score(y_test, y_pred_voting)
print(f"Voting Classifier Accuracy: {accuracy_voting:.2f}")
# %%
