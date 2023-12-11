# %%
#   Predict the purchase amount by age, gender, location, size, season, item purchased, and color.
#   What is the expected purchase amount for a customer's next transaction based on their previous purchasing behavior,
#   including the frequency of purchases, average review rating,and the types of items purchased?
# %%

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
preprocessed_data['Purchase Amount Group'] = pd.cut(
    preprocessed_data['Purchase Amount (USD)'], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
preprocessed_data['Purchase Amount Group'].value_counts()

# I created a new column that categorize person into 4 groups based on their purchase amount
preprocessed_data['Purchase Amount Group'].describe()
preprocessed_data['Purchase Amount Group'].value_counts().plot(kind='bar')


# %%
selected_columns = ['Age', 'Gender', 'Location',  'Item Purchased',
                    'Frequency of Purchases', 'Review Rating', 'Previous Purchases']

X = preprocessed_data[selected_columns]
y = preprocessed_data['Purchase Amount Group']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)


# Step 4: Train the KNN model
k = 10  # Adjust the value of k as needed
knn_model = KNeighborsClassifier(n_neighbors=k)
knn_model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = knn_model.predict(X_test)

# Step 6: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Classification report for more detailed evaluation
print(classification_report(y_test, y_pred))

# %%

# Assuming 'Target_Variable_Column_Name' is the column you want to predict
target_variable = 'Purchase Amount Group'

# Selecting features for clustering (customize based on your dataset)
selected_features = ['Age', 'Gender', 'Location', 'Size', 'Season', 'Item Purchased',
                     'Frequency of Purchases', 'Review Rating', 'Previous Purchases']

# Extracting the selected features
X = preprocessed_data[selected_features]

# Standardizing the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determining the optimal number of clusters using the Elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++',
                    max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plotting the Elbow method
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.show()

# Based on the Elbow method, choose the optimal number of clusters (let's say k=3)
optimal_clusters = 10

# Applying K-Means clustering
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++',
                max_iter=300, n_init=10, random_state=0)
preprocessed_data['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualizing the clusters (customize based on your dataset)
for cluster in range(optimal_clusters):
    cluster_data = preprocessed_data[preprocessed_data['Cluster'] == cluster]
    plt.scatter(
        cluster_data['Age'], cluster_data['Purchase Amount Group'], label=f'Cluster {cluster + 1}')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[
            :, 1], s=300, c='red', marker='*', label='Centroids')
plt.title('Customer Segmentation')
plt.xlabel('Gender')
plt.ylabel('Purchase Amount Group')
plt.legend()
plt.show()


# Visualizing the clusters using a pair plot
# cluster_palette = sns.color_palette("husl", optimal_clusters)
# sns.pairplot(preprocessed_data, hue='Cluster', palette=cluster_palette)
# plt.show()

# # Evaluate clustering using silhouette score
# silhouette_avg = silhouette_score(X_scaled, preprocessed_data['Cluster'])
# print(f"Silhouette Score: {silhouette_avg}")


# %%
# second try


X = preprocessed_data['Age', 'Gender', 'Location',
                      'Season']
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
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})

# Display coefficients sorted by magnitude
coefficients = coefficients.reindex(
    coefficients['Coefficient'].abs().sort_values(ascending=False).index)
print(coefficients)
# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
# accuracy = accuracy_score(y_test, y_pred)
# print(f'accuracy: {accuracy}')
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# %%
# Random Forest Regressor
# Import necessary libraries

# Separate features (X) and target variable (y)
X = preprocessed_data['Age', 'Subscription Status', 'Season',
                      'Frequency of Purchases', 'Review Rating', 'Previous Purchases']
y = preprocessed_data['Purchase Amount Group']

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
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

accuracy = accuracy_score(y_test, y_pred)
print(f'accuracy: {accuracy}')
# print(f'Mean Squared Error: {mse}')
# print(f'R-squared: {r2}')

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
