#Predicting Household Electricity Consumption (mini version of grid-scale demand forecasting)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,r2_score

df = pd.read_csv("ElectricityData.csv")
print(df.head())

df = df.dropna()
X = df[['hour']]
Y = df["consumption"]

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,shuffle=False)

model = LinearRegression()
model.fit(X_train,Y_train)

Y_pred = model.predict(X_test)

mae = mean_absolute_error(Y_test,Y_pred)
r2 = r2_score(Y_test,Y_pred)

print("Mean Absolute Error:",mae)
print("R2 Score:",r2)

plt.plot(Y_test.values,label="Actual")
plt.plot(Y_pred,label="Predicted")
plt.legend()
plt.title("Electricity Consumption: Actual vs Predicted")
plt.show()

