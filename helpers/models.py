import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Simple Feedforward NN for example to do **linear aproximation** of a Q-function."""

    def __init__(self, input_size: int, output_size: int, lr: float):
        super().__init__()
        # Define the neural network
        self.fc1 = nn.Linear(input_size, input_size // 2)
        self.fc2 = nn.Linear(input_size // 2, output_size)
        # Define a loss function
        self.criterion = nn.MSELoss()
        # Define an optimizer
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Given a state S returns a measure of each possible actions value.
        Returns logits as we want to preserve a sense of the total state value
        that would be always normalized if we were to softmax.
        """
        device = next(self.parameters()).device.type
        x = x.to(device)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def update(self, q_s, target) -> float:
        # Zero the gradients, perform a backward pass, and update the weights
        self.optimizer.zero_grad()
        loss = self.criterion(q_s, target)
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy().item()
