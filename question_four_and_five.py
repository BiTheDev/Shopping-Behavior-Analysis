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
# Can we predict the likelihood of a customer making a purchase in a specific season
# (e.g., Winter, Summer) based on their past purchase patterns and demographics?

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# Preprocessing: Creating target variables for seasonal purchases
data['Winter Purchase'] = (data['Season'] == 'Winter').astype(int)
data['Summer Purchase'] = (data['Season'] == 'Summer').astype(int)

# Feature selection
features = ['Age', 'Gender', 'Location', 'Previous Purchases', 'Review Rating', 'Subscription Status']
X = data[features]
y_winter = data['Winter Purchase']
y_summer = data['Summer Purchase']

# Data splitting for Winter and Summer
X_train_winter, X_test_winter, y_train_winter, y_test_winter = train_test_split(X, y_winter, test_size=0.2, random_state=42)
X_train_summer, X_test_summer, y_train_summer, y_test_summer = train_test_split(X, y_summer, test_size=0.2, random_state=42)

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

# Results
accuracy_winter, accuracy_summer, report_winter, report_summer


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

# Results
accuracy_winter_svc, accuracy_summer_svc, report_winter_svc, report_summer_svc
