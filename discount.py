
#%%
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import chi2_contingency 

# Load the dataset
file_path = './shopping_behavior_updated.csv'
data = pd.read_csv(file_path)

# Copying the dataset to avoid modifying the original data
preprocessed_data = data.copy()

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

# Displaying the first few rows of the preprocessed dataset
print(preprocessed_data.head())

# %%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report
# all vs. discount
preprocessed_data['Discount Applied'].equals(preprocessed_data['Promo Code Used']) # True
X_train, X_test, y_train, y_test = train_test_split(preprocessed_data.drop(['Discount Applied', 'Promo Code Used'], axis=1),
                                                    preprocessed_data['Discount Applied'], test_size=0.20, 
                                                    random_state=101)
knn = KNeighborsClassifier(n_neighbors=2) 
knn.fit(X_train,y_train)
print(f'knn train score:  {knn.score(X_train,y_train)}')
print(f'knn test score:  {knn.score(X_test,y_test)}')
print(confusion_matrix(y_test, knn.predict(X_test)))
print(classification_report(y_test, knn.predict(X_test)))
# %%
# all vs. discount
X_train, X_test, y_train, y_test = train_test_split(preprocessed_data.drop(['Discount Applied', 'Promo Code Used'], axis=1),
                                                    preprocessed_data['Discount Applied'], test_size=0.20, 
                                                    random_state=101)
# svc = SVC( gamma = 'auto') # 'scale', 'auto', 0.1, 5, 
# defining parameter range 
param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}  
  
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3) 
  
# fitting the model for grid search 
grid.fit(X_train, y_train) 
# svc.fit(X_train,y_train)
print(f'svc train score:  {grid.score(X_train,y_train)}')
print(f'svc test score:  {grid.score(X_test,y_test)}')
print(confusion_matrix(y_test, grid.predict(X_test)))
print(classification_report(y_test, grid.predict(X_test)))
# %%
# season vs. discount
X_train, X_test, y_train, y_test = train_test_split(preprocessed_data['Season'],
                                                    preprocessed_data['Discount Applied'], test_size=0.20, 
                                                    random_state=101)
X_train= X_train.values.reshape(-1, 1)
X_test = X_test.values.reshape(-1, 1)
lr = LogisticRegression()
lr.fit(X_train,y_train)
print(f'lr train score:  {lr.score(X_train,y_train)}')
print(f'lr test score:  {lr.score(X_test,y_test)}')
print(confusion_matrix(y_test, lr.predict(X_test)))
print(classification_report(y_test, lr.predict(X_test)))
# %%