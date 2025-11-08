import torch
import torch.nn as nn
import numpy as np

# ===========================================================
# Neural Network (same structure as in train_ai.py)
# ===========================================================
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

# ===========================================================
# AI Controller class
# ===========================================================
class AIController:
    def __init__(self, model_path="results/ai_model.pt"):
        self.model = NeuralCompensator()
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print("✅ Loaded AI model from", model_path)

    def compute(self, q_err_vec, w):
        # Convert inputs to a PyTorch tensor
        inp = np.concatenate([q_err_vec, w]).astype(np.float32)
        inp_t = torch.tensor(inp).unsqueeze(0)

        # Forward pass through the model
        with torch.no_grad():
            out = self.model(inp_t).cpu().numpy().flatten()

        # Apply tuned scale factor
        SCALE = -0.10  # <-- we’ll update this after scale search
        return SCALE * out









