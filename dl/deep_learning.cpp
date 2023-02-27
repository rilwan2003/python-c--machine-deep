#include <iostream>
#include <vector>
#include <dlib/dnn.h>

using namespace dlib;

int main() {
  // Define the deep neural network
  typedef loss_mean_squared<fc<1>> net_type;
  net_type net;

  // Train the neural network on input and output data
  std::vector<matrix<float>> samples = { {0.5, 1.0}, {1.0, 2.0}, {2.0, 4.0}, {4.0, 8.0} };
  std::vector<float> labels = { 1.0, 2.0, 4.0, 8.0 };
  dnn_trainer<net_type> trainer(net);
  trainer.set_learning_rate(0.01);
  trainer.set_min_learning_rate(1e-5);
  trainer.set_mini_batch_size(1);
  trainer.be_verbose();
  trainer.train(samples, labels);

  // Use the neural network to predict new values
  matrix<float> x_new = {0.7, 1.5};
  float y_pred = net(x_new);

  // Print the predicted value
  std::cout << "Predicted value: " << y_pred << std::endl;

  return 0;
}
