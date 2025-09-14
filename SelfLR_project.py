#Step 01 Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

#Step 02 Import Dataset
data = pd.read_csv("CO2 Emissions_Canada.csv")
print(data.head())

#Step 3 Exploration
print(data.info())
print(data.describe())
print(data.isnull().sum())

# #Step 04 Data Visualization
sns.scatterplot(x="Engine Size(L)",y="CO2 Emissions(g/km)",data=data)
plt.title("Engine Size vs CO2 Emissions")
plt.show()

numeric_data = data.select_dtypes(include=np.number)
sns.heatmap(numeric_data.corr(),annot=True,cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

#Step 05 features and target
X = data[["Engine Size(L)","Cylinders","Fuel Consumption Comb (mpg)"]]
Y = data["CO2 Emissions(g/km)"]

#Step 06 Splitting the Dataset
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

#Step 07 Training the Dataset
model = LinearRegression()
model.fit(X_train,Y_train)

#Step 08 Make Predictions
Y_pred = model.predict(X_test)

#Step 09 Evaluation
mse = mean_squared_error(Y_test,Y_pred)
r2 = r2_score(Y_test,Y_pred)

print("Mean Squared Error",mse)
print("r2 Score:",r2)

#Step 10 visualization Predictions
plt.scatter(Y_test, Y_pred, alpha=0.6)
plt.xlabel("Actual CO2 Emissions")
plt.ylabel("Predicted CO2 Emissions")
plt.title("Actual vs Predicted")
plt.show()
