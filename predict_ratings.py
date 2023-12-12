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
from sklearn.model_selection import train_test_split, cross_val_score, KFold
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
# Create linear regressor models
rfr = RandomForestRegressor(n_estimators=100, random_state=20)
lr = LinearRegression()
models = [rfr, lr]

# Train the models
for model in models:
    model.fit(X_train, y_train)

    # Make Predictions on the test set
    predictions = model.predict(X_test)

    # Evaluate Model Performance
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    explained_variance = explained_variance_score(y_test, predictions)
    
    # Get the model name
    model_name = model.__class__.__name__

    # Display the score results
    print(f"{model_name}: ")
    print(f"The Mean Squared Error is {mse:.2f}")
    print(f"The Mean Absolute Error is {mae:.2f} ")
    print(f"The R-squared is {r2:.2f}")
    print(f"The Explained Variance Score is {explained_variance: .2f}")
    print("----------")

    # Set up k-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=30)

    # Perform k-fold cross-validation
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')

    # Print the cross-validation results
    print(f'R-squared for each fold: {cv_scores}')
    print(f'Mean R-squared: {np.mean(cv_scores)}')
    print("***********************************************")

    metrics = ['MSE', 'MAE', 'R-squared', 'Explained Variance', 'K-fold CV']
    scores = [mse, mae, r2, explained_variance, cv_scores.mean()]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(metrics, scores, color=['blue', 'green', 'red', 'purple', 'orange'])

    # Adding the scores on top of the bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 3), ha='center', va='bottom')

    # Adding labels and title
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title(f'{model_name} Metrics')


# Display the plot
plt.show()

#%%
# Summerize:
# In both models, the R-squared values and the Explained Variance Scores
# are negative, indicating that the models are not performing well in 
# explaining the variance in the target variable. Additionally, the positive
# K-fold Cross-Validation mean R-squared for LinearRegression suggests a 
# slightly better performance compared to RandomForestRegressor, but overall, 
# the models may not be capturing the underlying patterns in the data 
# effectively.

#%%
# def decode(label_encoder, encoded_val):
#     if isinstance(encoded_val, float) or isinstance(encoded_val, int):
#         encoded_val = np.array([encoded_val])
#     return label_encoder.inverse_transform(encoded_val)

#%%
