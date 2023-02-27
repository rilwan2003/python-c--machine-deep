import torch

# Define the neural network architecture
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(2, 1)
    def forward(self, x):
        x = self.fc1(x)
        return x

net = Net()

# Train the neural network on input and output data
X = torch.Tensor([[0.5, 1.0], [1.0, 2.0], [2.0, 4.0], [4.0, 8.0]])
y = torch.Tensor([1.0, 2.0, 4.0, 8.0])
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
for epoch in range(1000):
    optimizer.zero_grad()
    y_pred = net(X)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()

# Use the neural network to predict new values
x_new = torch.Tensor([[0.7, 1.5]])
y_pred = net(x_new)

# Print the predicted value
print("Predicted value:", y_pred.item())
