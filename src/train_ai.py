import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import os

# ===============================
# Neural Network Definition
# ===============================
class NeuralCompensator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.net(x)

# ===============================
# Generate Synthetic Training Data
# ===============================
def generate_data(num_samples=50000):
    np.random.seed(42)
    # Inputs: attitude error (3), angular rate (3)
    X = np.random.uniform(-2, 2, (num_samples, 6))
    # Targets: desired torque (simulate nonlinear control response)
    Y = -0.12 * X[:, :3] - 0.06 * X[:, 3:] + 0.06 * np.tanh(2 * X[:, :3]) + 0.02 * np.sign(X[:, 3:])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

# ===============================
# Train the Neural Network
# ===============================
def train_model():
    model = NeuralCompensator()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    epochs = 200

    X, Y = generate_data()

    print(f"Training on {len(X)} samples for {epochs} epochs...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(X)

        # Improved loss with torque regularization for smoother control
        lam = 1e-3
        loss = ((out - Y)**2).mean() + lam * (out**2).mean()

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: loss={loss.item():.6f}")

    os.makedirs("results", exist_ok=True)
    torch.save(model.state_dict(), "results/ai_model.pt")
    print("✅ Model saved to results/ai_model.pt")

# ===============================
# Run the training
# ===============================
if __name__ == "__main__":
    train_model()
