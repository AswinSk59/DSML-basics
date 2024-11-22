import numpy as np
import matplotlib.pyplot as plt import pandas as pd
from sklearn.model_selection import train_test_split from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
dataset = pd.read_csv('/content/Salary_data.csv') X = dataset.iloc[:, :-1].values
 
y = dataset.iloc[:, 1].values print(dataset.head())


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)
plt.scatter(X_train, y_train, color='red', label='Actual')
plt.plot(X_train, regressor.predict(X_train), color='blue', label='Predicted') plt.title('Salary VS Experience (Training set)')
plt.xlabel('Year of Experience') plt.ylabel('Salary')
plt.legend() # Add a legend to distinguish between actual and predicted data plt.show()
plt.scatter(X_test, y_test, color='red', label='Actual')
plt.plot(X_train, regressor.predict(X_train), color='blue', label='Predicted') plt.title('Salary VS Experience (Test set)')
plt.xlabel('Year of Experience') plt.ylabel('Salary')
plt.legend() # Add a legend to distinguish between actual and predicted data plt.show()
regressor = LinearRegression() regressor.fit(X_train, y_train)


y_pred = regressor.predict(X_test)
 
y_pred

mse = mean_squared_error(y_test, y_pred) rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred) r2 = r2_score(y_test, y_pred)
print("Mean Squared Error (MSE):", mse) print("Root Mean Squared Error (RMSE):", rmse) print("Mean Absolute Error (MAE):", mae) print("R-squared (R2):", r2)
new_input = [[5]]
y_pred = regressor.predict(new_input) print("Predicted Salary:", y_pred)
