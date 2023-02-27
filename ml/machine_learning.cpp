#include <iostream>
#include <vector>
#include <mlpack/methods/linear_regression/linear_regression.hpp>

int main() {
  // Define the input and output data
  arma::mat X = { {0.5, 1.0}, {1.0, 2.0}, {2.0, 4.0}, {4.0, 8.0} };
  arma::vec y = { 1.0, 2.0, 4.0, 8.0 };

  // Train the linear regression model
  mlpack::regression::LinearRegression lr(X, y);

  // Use the model to predict new values
  arma::vec x_new = { 0.7, 1.5 };
  arma::vec y_pred;
  lr.Predict(x_new, y_pred);

  // Print the predicted values
  std::cout << "Predicted values: " << y_pred << std::endl;

  return 0;
}
