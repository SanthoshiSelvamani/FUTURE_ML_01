import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
data = pd.read_csv("sales_dataset.csv")
data['Date'] = pd.to_datetime(data['Date'])
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day
X = data[['Year','Month','Day']]
y = data['Sales']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
model = LinearRegression()
model.fit(X_train,y_train)

# Prediction
predictions = model.predict(X_test)

# Plot forecast
plt.figure(figsize=(10,5))

plt.plot(y_test.values, label="Actual Sales")
plt.plot(predictions, label="Predicted Sales")

plt.title("Sales Demand Forecasting")
plt.xlabel("Time")
plt.ylabel("Sales")

plt.legend()

plt.show()