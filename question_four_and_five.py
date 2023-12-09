#%% Preprocess the data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
file_path = './shopping_behavior_updated.csv'
data = pd.read_csv(file_path)

# Preprocessing the data as per the provided script
preprocessed_data = data.copy()

# Encoding categorical variables using LabelEncoder
label_encoders = {}
categorical_columns = ['Gender', 'Item Purchased', 'Category', 'Location', 'Size', 'Color', 
                       'Season', 'Subscription Status', 'Shipping Type', 'Discount Applied', 
                       'Promo Code Used', 'Payment Method', 'Frequency of Purchases']

for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    preprocessed_data[column] = label_encoders[column].fit_transform(preprocessed_data[column])

# Normalizing numerical variables
numerical_columns = ['Age', 'Purchase Amount (USD)', 'Review Rating', 'Previous Purchases']
scaler = StandardScaler()
preprocessed_data[numerical_columns] = scaler.fit_transform(preprocessed_data[numerical_columns])

# Creating a binary target variable for 'Clothing' category (assuming 'Clothing' is encoded as 1)
preprocessed_data['Is_Clothing'] = (preprocessed_data['Category'] == 1).astype(int)


#%% Question 4
#  What age group constitutes the primary force for purchasing items in the ‘Clothing’ category? (Using K-Nearest Neighbors)
# Selecting the features and target variable for the KNN model
features = ['Age', 'Gender', 'Previous Purchases']
target = 'Is_Clothing'

X_knn = preprocessed_data[features]
y_knn = preprocessed_data[target]
X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(X_knn, y_knn, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_knn, y_train_knn)
y_pred_knn = knn.predict(X_test_knn)

accuracy_knn = accuracy_score(y_test_knn, y_pred_knn)
report_knn = classification_report(y_test_knn, y_pred_knn)

print("K-Nearest Neighbors Model Results:")
print("Accuracy:", accuracy_knn)
print("Classification Report:\n", report_knn)



#%% Question 5
# Can we predict the likelihood of a customer making a purchase in a specific category based on demographics
#  (Age, Gender), location, and previous purchase history? (Using Logistic Regression and Random Forests)


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

features_lr_rf = ['Age', 'Gender', 'Location', 'Previous Purchases']
target_lr_rf = 'Category'

X_lr_rf = preprocessed_data[features_lr_rf]
y_lr_rf = preprocessed_data[target_lr_rf]
X_train_lr_rf, X_test_lr_rf, y_train_lr_rf, y_test_lr_rf = train_test_split(X_lr_rf, y_lr_rf, test_size=0.2, random_state=42)

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_lr_rf, y_train_lr_rf)
y_pred_log_reg = log_reg.predict(X_test_lr_rf)

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train_lr_rf, y_train_lr_rf)
y_pred_rf = random_forest.predict(X_test_lr_rf)

accuracy_log_reg = accuracy_score(y_test_lr_rf, y_pred_log_reg)
report_log_reg = classification_report(y_test_lr_rf, y_pred_log_reg)
accuracy_rf = accuracy_score(y_test_lr_rf, y_pred_rf)
report_rf = classification_report(y_test_lr_rf, y_pred_rf)

print("\nLogistic Regression Model Results:")
print("Accuracy:", accuracy_log_reg)
print("Classification Report:\n", report_log_reg)

print("\nRandom Forest Model Results:")
print("Accuracy:", accuracy_rf)
print("Classification Report:\n", report_rf)



# %%
