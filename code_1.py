import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# Load the dataset
url = 'https://raw.githubusercontent.com/BiTheDev/Shopping-Behavior-Analysis/main/shopping_behavior_updated.csv'
data = pd.read_csv(url)

# Copying the dataset to avoid modifying the original data
preprocessed_data = data.copy()

sns.histplot(data=preprocessed_data, x="Age")
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()
#%%
sns.histplot(data=preprocessed_data, x="Age", hue="Gender", multiple="dodge", shrink=.8)
plt.title("Age Distribution by Gender")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()
#%%
sns.histplot(data=preprocessed_data, x="Category", shrink=.8)
plt.title("Purchased Item Category Distribution")
plt.xlabel("Category")
plt.ylabel("Count")
plt.show()
#%%
sns.histplot(data=preprocessed_data, x="Season", shrink=.8)
plt.title("Shopping Season Distribution")
plt.xlabel("Season")
plt.ylabel("Count of purchase")
plt.show()
#%%
sns.histplot(data=preprocessed_data, x="Frequency of Purchases", shrink=.8)
plt.title("Frequency of Purchases Distribution")
plt.xlabel("Frequency")
plt.xticks(rotation = 45)
plt.ylabel("Count")
plt.show()
#%%
arrFall=preprocessed_data['Season'] == 'Fall'
arrSpring=preprocessed_data['Season'] == 'Spring'
arrSummer=preprocessed_data['Season'] == 'Summer'
arrWinter=preprocessed_data['Season'] == 'Winter'

vFall = preprocessed_data[arrFall]
vSpring = preprocessed_data[arrSpring]
vSummer = preprocessed_data[arrSummer]
vWinter= preprocessed_data[arrWinter]

fallSales = vFall['Purchase Amount (USD)'].sum()
springSales = vSpring['Purchase Amount (USD)'].sum()
summerSales = vSummer['Purchase Amount (USD)'].sum()
winterSales = vWinter['Purchase Amount (USD)'].sum()

fallDiscounts = vFall['Discount Applied'].value_counts()['Yes']
springDiscounts = vSpring['Discount Applied'].value_counts()['Yes']
summerDiscounts = vSummer['Discount Applied'].value_counts()['Yes']
winterDiscounts = vWinter['Discount Applied'].value_counts()['Yes']

fallDiscountsN = vFall['Discount Applied'].value_counts()['No']
springDiscountsN = vSpring['Discount Applied'].value_counts()['No']
summerDiscountsN = vSummer['Discount Applied'].value_counts()['No']
winterDiscountsN = vWinter['Discount Applied'].value_counts()['No']

data = [['Winter', winterSales, winterDiscounts], ['Spring', springSales, springDiscounts],
        ['Summer', summerSales, summerDiscounts], ['Fall', fallSales, fallDiscounts]]
sales = pd.DataFrame(data, columns=['Season', 'Sales', 'Discount count'])

chi_data = [[winterDiscounts.astype('int'), springDiscounts.astype('int'), summerDiscounts.astype('int'), fallDiscounts.astype('int')],
            [winterDiscountsN.astype('int'), springDiscountsN.astype('int'), summerDiscountsN.astype('int'), fallDiscountsN.astype('int')]]
stat, p, dof, expected = chi2_contingency(chi_data)
chi_data = pd.DataFrame(chi_data)
chi_data.columns= ['Winter', 'Spring', 'Summer','Fall']
chi_data.index = ['Discount used', 'Discount not used']
print(pd.DataFrame(chi_data))
significance_level = 0.05
print("p value: " + str(p))
if p <= significance_level:
    print('Reject NULL HYPOTHESIS')
else:
    print('ACCEPT NULL HYPOTHESIS')
sales.plot(x='Season', y='Sales', kind='bar', legend=False)
plt.xlabel('Season')
plt.xticks(rotation = 0)
plt.ylabel('Sales in $')
plt.title('Best Season Histogram')
plt.show()
#%%
sales.plot(x='Season', y='Discount count', kind='bar', legend=False)
plt.xlabel('Season')
plt.xticks(rotation = 0)
plt.ylabel('Number of discount used')
plt.title('Discount Histogram')
plt.show()
#%%

#%% Question: Can we assess the impact of the season, alongside demographics and past purchase behavior, on the likelihood of making a purchase within the clothing category?
url = 'https://raw.githubusercontent.com/BiTheDev/Shopping-Behavior-Analysis/main/shopping_behavior_updated.csv'
data = pd.read_csv(url)
# One-hot encode categorical features including 'Season'
categorical_features = ['Gender', 'Location', 'Subscription Status', 'Season', 'Shipping Type', 'Discount Applied', 'Promo Code Used', 'Payment Method', 'Frequency of Purchases']
numerical_features = ['Age', 'Previous Purchases', 'Review Rating']

preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(), categorical_features),
    ('num', 'passthrough', numerical_features)
])

# Assuming that we are predicting a purchase within the 'Clothing' category
y = (data['Category'] == 'Clothing').astype(int)
X = data.drop(columns=['Customer ID', 'Item Purchased', 'Category', 'Purchase Amount (USD)', 'Size', 'Color'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the RandomForestClassifier within a Pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])
pipeline.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Display results
print(f"Accuracy for predicting clothing purchases: {accuracy}")
print("Classification report: \n", report)

# Feature importance
feature_importances = pipeline.named_steps['classifier'].feature_importances_
feature_names = preprocessor.transformers_[0][1].get_feature_names_out(categorical_features)
feature_names = np.concatenate([feature_names, numerical_features])

# Combine feature importances with feature names
feature_importance_dict = dict(zip(feature_names, feature_importances))

# Sort the feature importances in descending order and print
sorted_feature_importances = sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True)
print("Feature importances: ", sorted_feature_importances)


# Define the SVC pipeline
pipeline_svc = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC(random_state=42))
])

# Parameter grid for GridSearchCV
param_grid = {
    'classifier__C': [0.1, 1, 10],
    'classifier__kernel': ['linear', 'rbf'],
    'classifier__gamma': ['scale', 'auto']
}

# Setup GridSearchCV
grid_search = GridSearchCV(pipeline_svc, param_grid, cv=5, scoring='accuracy')

# Fit the model
grid_search.fit(X_train, y_train)

# Get the best estimator
best_svc = grid_search.best_estimator_

# Predictions
y_pred_svc = best_svc.predict(X_test)

# Evaluation
accuracy_svc = accuracy_score(y_test, y_pred_svc)
report_svc = classification_report(y_test, y_pred_svc)

# Display results
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best SVC accuracy: {accuracy_svc}")
print("SVC Classification report: \n", report_svc)


#%% hyperparameter tuning

# One-hot encode categorical features including 'Season'
categorical_features = ['Gender', 'Location', 'Subscription Status', 'Season', 'Shipping Type', 'Discount Applied', 'Promo Code Used', 'Payment Method', 'Frequency of Purchases']
numerical_features = ['Age', 'Previous Purchases', 'Review Rating']

preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(), categorical_features),
    ('num', 'passthrough', numerical_features)
])

# Assuming that we are predicting a purchase within the 'Clothing' category
y = (data['Category'] == 'Clothing').astype(int)
X = data.drop(columns=['Customer ID', 'Item Purchased', 'Category', 'Purchase Amount (USD)', 'Size', 'Color'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVC pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('svc', SVC(random_state=42))
])

# Parameter grid for GridSearch
param_grid = {
    'svc__C': [0.1, 1, 10],
    'svc__gamma': [1, 0.1, 0.01],
    'svc__kernel': ['rbf', 'linear']
}

# Grid search with cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best parameters found
print("Best parameters found: ", grid_search.best_params_)

# Best model
best_model = grid_search.best_estimator_

# Predict and evaluate the model
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Display results
print(f"Accuracy for predicting clothing purchases with tuned SVC: {accuracy}")
print("Classification report with tuned SVC: \n", report)


#%% Number of Clothing Purchases by Age Group

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



#%% Gender Distribution in Clothing Purchases

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


