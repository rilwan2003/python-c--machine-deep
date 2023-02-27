from sklearn.linear_model import LinearRegression

# Define the input and output data
X = [[0.5, 1.0], [1.0, 2.0], [2.0, 4.0], [4.0, 8.0]]
y = [1.0, 2.0, 4.0, 8.0]

# Train the linear regression model
lr = LinearRegression().fit(X, y)

# Use the model to predict new values
x_new = [[0.7, 1.5]]
y_pred = lr.predict(x_new)

# Print the predicted values
print("Predicted values:", y_pred)
