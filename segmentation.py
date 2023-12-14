
# %%
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
from sklearn.metrics import accuracy_score, confusion_matrix
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
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# preprocessing the data
file_path = './shopping_behavior_updated.csv'
data = pd.read_csv(file_path)

# Copying the dataset to avoid modifying the original data
preprocessed_data = data.copy()

# Encoding categorical variables using LabelEncoder
label_encoders = {}
print(preprocessed_data.head())
print(preprocessed_data["Category"].value_counts())

categorical_columns = ['Gender', 'Item Purchased', 'Category', 'Location', 'Size', 'Color',
                       'Season', 'Subscription Status', 'Shipping Type', 'Discount Applied',
                       'Promo Code Used', 'Payment Method', 'Frequency of Purchases']

for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    preprocessed_data[column] = label_encoders[column].fit_transform(
        preprocessed_data[column])

for column, encoder in label_encoders.items():
    print(f"Column: {column}")
    print(f"Original Values: {encoder.classes_}")
    print(f"Encoded Labels: {encoder.transform(encoder.classes_)}")
    print()
preprocessed_data.fillna(0, inplace=True)
# Normalizing numerical variables
numerical_columns = [
    'Review Rating', 'Previous Purchases']
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

# %%
preprocessed_data.head(5)
preprocessed_data.isna().sum()

# %%
df = preprocessed_data.copy()
df_segments = df[["Age", "Gender", "Season", "Subscription Status", "Category",
                  "Purchase Amount (USD)", "Frequency of Purchases"]]

# Initialize an empty list to store the Within-Cluster-Sum-of-Squares (WCSS) for different cluster numbers
wcss = []
plt.figure(figsize=(6, 4))
df['Gender'].value_counts().plot(kind='bar', color=['blue', 'pink'])
plt.title('Distribution of Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()
# Iterate over different numbers of clusters to find the optimal number
for i in range(1, 8):
    # Create a KMeans instance with the current number of clusters
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(df)  # Fit the model to the data
    wcss_iter = kmeans.inertia_  # Get the WCSS for the current number of clusters
    wcss.append(wcss_iter)  # Append the WCSS to the list

# Create a line plot to visualize the Elbow Method
number_clusters = range(1, 8)
plt.plot(number_clusters, wcss)
plt.title('The Elbow Method')  # Set the title of the plot
plt.xlabel('Number of clusters')  # Set the label for the X-axis
plt.ylabel('WCSS')  # Set the label for the Y-axis

# Display the plot
plt.show()

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(df_segments)
cluster_labels = kmeans.predict(df_segments)
df_segments["Cluster"] = cluster_labels

df_segments
plt.scatter(
    df_segments[['Age']],  # X-axis: Age
    df_segments['Purchase Amount (USD)'],  # Y-axis: Purchase Amount (USD)
    c=kmeans.labels_,  # Color points by cluster labels assigned by KMeans
    cmap='viridis'  # Specify a color map for better visualization
)

plt.title('Customer Segments Using K-Means')
plt.xlabel('Age')
plt.ylabel('Purchase Amount (USD)')
plt.show()


# %%

# Assuming 'df_segments' is your DataFrame with clustered data

# Create a 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
print(df_segments.columns)
# Scatter plot for Age, Purchase Amount (USD), and Frequency of Purchases
scatter = ax.scatter(
    df_segments['Age'],
    df_segments['Purchase Amount (USD)'],
    df_segments['Frequency of Purchases'],
    c=df_segments['Cluster'],
    cmap='viridis',
    marker='o',  # Set marker style
    s=50  # Set marker size
)

# Customize plot labels
ax.set_xlabel('Age')
ax.set_ylabel('Purchase Amount (USD)')
ax.set_zlabel('Frequency of Purchases')
ax.set_title('Customer Segments Using K-Means (3D)')

# Add a color bar to the right of the plot
colorbar = plt.colorbar(scatter)
colorbar.set_label('Cluster')

# Show the plot
plt.show()


# Assuming df_segments contains the one-hot encoded columns, including 'Gender_0' and 'Gender_1'


# %%
# plt.scatter(
#     df_segments['Gender'],  # X-axis: Age
#     df_segments['Purchase Amount (USD)'],  # Y-axis: Purchase Amount (USD)
#     c=kmeans.labels_,  # Color points by cluster labels assigned by KMeans
#     cmap='viridis'  # Specify a color map for better visualization
# )

# plt.title('Customer Segments Using K-Means')
# plt.xlabel('Gender')
# plt.ylabel('Purchase Amount (USD)')
# plt.show()
# %%

# Assuming you already have preprocessed_data and df_segments

# Split data into features (X) and target variable (y)
X = df  # Features
y = df_segments["Cluster"]  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Standardize the features (important for KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Instantiate the KNN classifier (you can choose the number of neighbors)
knn = KNeighborsClassifier(n_neighbors=24)

# Fit the model to the training data
knn.fit(X_train_scaled, y_train)

# Predictions on the test set
y_pred = knn.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)

# Visualization (scatter plot might not be suitable for classification)
# You may want to use other visualization techniques, like a confusion matrix heatmap

# Note: You can adjust the number of neighbors and other parameters based on your needs

# %%
# Split data into features (X) and target variable (y)
X = df_segments[["Age", "Purchase Amount (USD)"]]  # Features
y = df_segments["Cluster"]  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Standardize the features (important for KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the KNN classifier
knn = KNeighborsClassifier()

# Define the hyperparameters to tune
# You can adjust the range based on your requirements
param_grid = {'n_neighbors': np.arange(1, 31)}

# Perform GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

# Print the best hyperparameters
best_params = grid_search.best_params_
print(f"Best Hyperparameters: {best_params}")

# Evaluate the model with the best hyperparameters on the test set
best_knn = grid_search.best_estimator_
y_pred = best_knn.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on Test Set: {accuracy}")
best_knn = model


# %%
# Can we predict the likelihood of a customer making a purchase in a specific category
# (like 'Clothing' or 'Footwear') based on their demographics (Age, Gender), location,
# and previous purchase history?

print(preprocessed_data.head())
# %%

# can we predict the category of the item purchased based on
