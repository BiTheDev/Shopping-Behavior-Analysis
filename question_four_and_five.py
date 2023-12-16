#%% [Data Loading and Initial Exploration]

import pandas as pd

# Load and display the first few rows of the dataset
file_path = './shopping_behavior_updated.csv'
data = pd.read_csv(file_path)
data.head()


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
data['Fall Purchase'] = (data['Season'] == 'Fall').astype(int)

# Feature selection
features = ['Age', 'Gender', 'Location', 'Previous Purchases', 'Review Rating', 'Subscription Status']
X = data[features]
y_winter = data['Winter Purchase']
y_summer = data['Summer Purchase']
y_spring = data['Spring Purchase']
y_fall = data['Fall Purchase']

# Data splitting for each season
X_train_winter, X_test_winter, y_train_winter, y_test_winter = train_test_split(X, y_winter, test_size=0.2, random_state=42)
X_train_summer, X_test_summer, y_train_summer, y_test_summer = train_test_split(X, y_summer, test_size=0.2, random_state=42)
X_train_spring, X_test_spring, y_train_spring, y_test_spring = train_test_split(X, y_spring, test_size=0.2, random_state=42)
X_train_fall, X_test_fall, y_train_fall, y_test_fall = train_test_split(X, y_fall, test_size=0.2, random_state=42)

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
rf_model.fit(X_train_fall, y_train_fall)
y_pred_fall = rf_model.predict(X_test_fall)
accuracy_fall = accuracy_score(y_test_fall, y_pred_fall)
report_fall= classification_report(y_test_fall, y_pred_fall)

# Results


print("Accuracy for winter purchase: ", accuracy_winter)
print("Report for winter purchase: \n", report_winter)
print("Accuracy for summer purchase: ", accuracy_summer)
print("Report for summer purchase: \n", report_summer)
print("Accuracy for spring purchase: ", accuracy_spring)
print("Report for spring purchase: \n", report_spring)
print("Accuracy for fall purchase: ", accuracy_fall)
print("Report for fall purchase: \n", report_fall)



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

accuracy_fall_svc, report_fall_svc = train_evaluate_svc(X_train_fall, X_test_fall, y_train_fall, y_test_fall)


# Results
accuracy_winter_svc, accuracy_summer_svc, report_winter_svc, report_summer_svc, accuracy_spring_svc, report_spring_svc, accuracy_winter_svc, report_winter_svc


print("SVC Accuracy for winter purchase: ", accuracy_winter_svc)
print("SVC Report for winter purchase: \n", report_winter_svc)
print("SVC Accuracy for summer purchase: ", accuracy_summer_svc)
print("SVC Report for summer purchase: \n", report_summer_svc)
print("SVC Accuracy for spring purchase: ", accuracy_spring_svc)
print("SVC Report for spring purchase: \n", report_spring_svc)
print("SVC Accuracy for fall purchase: ", accuracy_fall_svc)
print("SVC Report for fall purchase: \n", report_fall_svc)





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
