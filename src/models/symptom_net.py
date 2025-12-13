import torch
import torch.nn as nn

class SymptomNet(nn.Module):
    def __init__(self, input_size=132, hidden_size=64, num_classes=41, dropout_rate=0.5):
        super(SymptomNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.layer2(out)
        return out

if __name__ == "__main__":
    # Smoke test
    model = SymptomNet()
    print(model)
    dummy_input = torch.randn(1, 132)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
