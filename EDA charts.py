#%%
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = './shopping_behavior_updated.csv'
data = pd.read_csv(file_path)

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