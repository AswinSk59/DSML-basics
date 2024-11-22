import pandas as pd import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
data_df = pd.read_excel('/content/CCCP.xlsx') data_df.head()
 
 

x = data_df.drop(['PE'], axis=1).values print(x)
y = data_df['PE'].values print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)
regressor = LinearRegression() regressor.fit(x_train, y_train)
 

y_pred = regressor.predict(x_test) print(y_pred)
plt.figure(figsize=(15, 10)) plt.scatter(y_test, y_pred)
 
plt.xlabel('Actual') plt.ylabel('Predicted') plt.title('ACTUAL VS PREDICTED')
plt.show()
mse = mean_squared_error(y_test, y_pred)rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)r2 = r2_score(y_test, y_pred) print("Mean Squared Error (MSE):", mse) print("Root Mean Squared Error (RMSE):", rmse)print("Mean Absolute Error (MAE):", mae) print("R-squared (R2):", r2)

new_input = [[14.96, 41.76, 1024.07, 73.17]]
y_pred = regressor.predict(new_input) print("Predicted target value:", y_pred)
