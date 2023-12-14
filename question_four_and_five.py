#%% [Data Loading and Initial Exploration]

import pandas as pd

# Load and display the first few rows of the dataset
file_path = './shopping_behavior_updated.csv'
data = pd.read_csv(file_path)
data.head()


#%% [Analyzing Clothing Purchases by Age Group]

import matplotlib.pyplot as plt
import seaborn as sns

# Filter data for 'Clothing' category and group by age
clothing_data = data[data['Category'] == 'Clothing']
age_group_counts = clothing_data.groupby('Age').size()

# Plot the distribution of clothing purchases by age
plt.figure(figsize=(12, 6))
sns.barplot(x=age_group_counts.index, y=age_group_counts.values)
plt.title('Number of Clothing Purchases by Age Group')
plt.xlabel('Age')
plt.ylabel('Number of Purchases')
plt.xticks(rotation=45)
plt.show()


#%% [Seasonal Purchase Prediction Model]
# Can we predict the likelihood of a customer making a purchase in each season
# based on their past purchase patterns and demographics?

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# Preprocessing: Creating target variables for seasonal purchases
data['Winter Purchase'] = (data['Season'] == 'Winter').astype(int)
data['Summer Purchase'] = (data['Season'] == 'Summer').astype(int)
data['Spring Purchase'] = (data['Season'] == 'Spring').astype(int)
data['Winter Purchase'] = (data['Season'] == 'Winter').astype(int)

# Feature selection
features = ['Age', 'Gender', 'Location', 'Previous Purchases', 'Review Rating', 'Subscription Status']
X = data[features]
y_winter = data['Winter Purchase']
y_summer = data['Summer Purchase']
y_spring = data['Spring Purchase']
y_winter = data['Winter Purchase']

# Data splitting for each season
X_train_winter, X_test_winter, y_train_winter, y_test_winter = train_test_split(X, y_winter, test_size=0.2, random_state=42)
X_train_summer, X_test_summer, y_train_summer, y_test_summer = train_test_split(X, y_summer, test_size=0.2, random_state=42)
X_train_spring, X_test_spring, y_train_spring, y_test_spring = train_test_split(X, y_spring, test_size=0.2, random_state=42)
X_train_winter, X_test_winter, y_train_winter, y_test_winter = train_test_split(X, y_winter, test_size=0.2, random_state=42)

# Pipeline with OneHotEncoder and RandomForestClassifier
categorical_features = ['Gender', 'Location', 'Subscription Status']
one_hot_encoder = OneHotEncoder()
preprocessor = ColumnTransformer(transformers=[('cat', one_hot_encoder, categorical_features)], remainder='passthrough')
rf_model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(random_state=42))])

# Train and evaluate the model for Winter Purchases
rf_model.fit(X_train_winter, y_train_winter)
y_pred_winter = rf_model.predict(X_test_winter)
accuracy_winter = accuracy_score(y_test_winter, y_pred_winter)
report_winter = classification_report(y_test_winter, y_pred_winter)

# Train and evaluate the model for Summer Purchases
rf_model.fit(X_train_summer, y_train_summer)
y_pred_summer = rf_model.predict(X_test_summer)
accuracy_summer = accuracy_score(y_test_summer, y_pred_summer)
report_summer = classification_report(y_test_summer, y_pred_summer)

# Train and evaluate the model for Spring Purchases
rf_model.fit(X_train_spring, y_train_spring)
y_pred_spring = rf_model.predict(X_test_spring)
accuracy_spring = accuracy_score(y_test_summer, y_pred_spring)
report_spring = classification_report(y_test_spring, y_pred_spring)

# Train and evaluate the model for Winter Purchases
rf_model.fit(X_train_winter, y_train_winter)
y_pred_winter = rf_model.predict(X_test_winter)
accuracy_winter = accuracy_score(y_test_winter, y_pred_winter)
report_winter = classification_report(y_test_winter, y_pred_winter)

# Results
accuracy_winter, accuracy_summer, report_winter, report_summer, accuracy_spring, report_spring, accuracy_winter,report_winter


#%% [SVC Model for Seasonal Purchase Prediction]

from sklearn.svm import SVC

# Function to train and evaluate SVC model
def train_evaluate_svc(X_train, X_test, y_train, y_test):
    svc_model = Pipeline(steps=[
        ('preprocessor', preprocessor), 
        ('classifier', SVC(random_state=42, kernel='rbf', C=1.0))
    ])
    svc_model.fit(X_train, y_train)
    y_pred = svc_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report

# Train and evaluate for Winter Purchases
accuracy_winter_svc, report_winter_svc = train_evaluate_svc(X_train_winter, X_test_winter, y_train_winter, y_test_winter)

# Train and evaluate for Summer Purchases
accuracy_summer_svc, report_summer_svc = train_evaluate_svc(X_train_summer, X_test_summer, y_train_summer, y_test_summer)


# Train and evaluate for Spring Purchases
accuracy_spring_svc, report_spring_svc = train_evaluate_svc(X_train_spring, X_test_spring, y_train_spring, y_test_spring)


# Train and evaluate for Winter Purchases

accuracy_winter_svc, report_winter_svc = train_evaluate_svc(X_train_winter, X_test_winter, y_train_winter, y_test_winter)


# Results
accuracy_winter_svc, accuracy_summer_svc, report_winter_svc, report_summer_svc, accuracy_spring_svc, report_spring_svc, accuracy_winter_svc, report_winter_svc

# %%

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
file_path = './shopping_behavior_updated.csv'
data = pd.read_csv(file_path)

# Encoding categorical variables
categorical_columns = ['Gender', 'Location', 'Subscription Status', 'Shipping Type', 'Discount Applied', 'Promo Code Used', 'Payment Method']
for column in categorical_columns:
    data[column] = LabelEncoder().fit_transform(data[column])

# Mapping 'Season' to numeric values
season_mapping = {season: idx for idx, season in enumerate(data['Season'].unique())}
data['Season'] = data['Season'].map(season_mapping)

# Creating a target variable for not purchasing in Spring (assuming Spring is mapped to a specific index)
data['No Purchase in Spring'] = (data['Season'] != season_mapping['Spring']).astype(int)

# Selecting relevant features
features = ['Age', 'Gender', 'Location', 'Previous Purchases']
X = data[features]
y = data['No Purchase in Spring']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Predicting and evaluating the model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

# Output the accuracy
print("Accuracy of the model:", accuracy)

# %%

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# Load the dataset
file_path = './shopping_behavior_updated.csv'
data = pd.read_csv(file_path)

# Encoding categorical variables
categorical_columns = ['Gender', 'Location', 'Subscription Status', 'Shipping Type', 'Discount Applied', 'Promo Code Used', 'Payment Method']
for column in categorical_columns:
    data[column] = LabelEncoder().fit_transform(data[column])

# Mapping 'Season' to numeric values
season_mapping = {season: idx for idx, season in enumerate(data['Season'].unique())}
data['Season'] = data['Season'].map(season_mapping)

# Creating a target variable for not purchasing in Spring
data['No Purchase in Spring'] = (data['Season'] != season_mapping['Spring']).astype(int)

# Selecting relevant features
features = ['Age', 'Gender', 'Location', 'Previous Purchases']
X = data[features]
y = data['No Purchase in Spring']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initializing the SVC model
svc = SVC(random_state=42)

# Defining the parameter grid for fine-tuning
param_grid = {
    'C': [0.1, 1, 10, 100],  # Regularization parameter
    'gamma': ['scale', 'auto'],  # Kernel coefficient
    'kernel': ['linear', 'rbf', 'poly']  # Specifies the kernel type to be used in the algorithm
}

# Setting up GridSearchCV to find the best parameters
grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, cv=5, verbose=2, n_jobs=-1)

# Fitting GridSearchCV to the training data
grid_search.fit(X_train_scaled, y_train)

# Extracting the best parameters and model
best_params = grid_search.best_params_
best_svc = grid_search.best_estimator_

# Predicting with the best model
y_pred_best = best_svc.predict(X_test_scaled)

# Evaluating the best model
accuracy_best = accuracy_score(y_test, y_pred_best)

# Outputting the best parameters and the accuracy of the best model
print("Best Parameters:", best_params)
print("Accuracy of the best SVC model:", accuracy_best)

# %% Gender Distribution in Clothing Purchases


import matplotlib.pyplot as plt
import seaborn as sns


clothing_data = data[data['Category'] == 'Clothing']
gender_counts = clothing_data['Gender'].value_counts()

plt.figure(figsize=(8, 5))
sns.barplot(x=gender_counts.index, y=gender_counts.values)
plt.title('Gender Distribution in Clothing Purchases')
plt.xlabel('Gender')
plt.ylabel('Number of Purchases')
plt.show()


#%% Purchase Amount Distribution in Clothing Category

plt.figure(figsize=(8, 5))
sns.histplot(clothing_data['Purchase Amount (USD)'], bins=20, kde=True)
plt.title('Purchase Amount Distribution in Clothing Category')
plt.xlabel('Purchase Amount (USD)')
plt.ylabel('Frequency')
plt.show()


#%% Purchase Frequency by Location for Clothing Category
location_counts = clothing_data['Location'].value_counts()
plt.figure(figsize=(12, 6))
sns.barplot(x=location_counts.index, y=location_counts.values)
plt.title('Purchase Frequency by Location for Clothing Category')
plt.xlabel('Location')
plt.ylabel('Number of Purchases')
plt.xticks(rotation=90)
plt.show()


#%% Previous Purchases vs Clothing Purchases

plt.figure(figsize=(8, 5))
sns.scatterplot(data=clothing_data, x='Previous Purchases', y='Purchase Amount (USD)', hue='Gender')
plt.title('Previous Purchases vs Clothing Purchases')
plt.xlabel('Previous Purchases')
plt.ylabel('Purchase Amount (USD)')
plt.legend(title='Gender')
plt.show()

# %%
