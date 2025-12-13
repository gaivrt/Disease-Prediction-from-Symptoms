import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add project root to path to ensure imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.data.loader import DataLoader
from src.models.symptom_net import SymptomNet

def train_model(num_epochs=50, learning_rate=0.001, batch_size=32):
    # 1. Prepare Data
    loader = DataLoader()
    X_train, y_train, X_test, y_test, encoder = loader.load_data()
    
    # Convert to Tensors
    X_train_tensor = torch.FloatTensor(X_train.values)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test.values)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Create DataLoaders
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    
    # 2. Initialize Model
    input_size = X_train.shape[1]
    num_classes = len(encoder.classes_)
    
    model = SymptomNet(input_size=input_size, num_classes=num_classes)
    
    # 3. Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 4. Training Loop
    train_losses = []
    val_losses = []
    
    print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for i, (features, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        # Calculate Validation Loss
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test_tensor)
            val_loss = criterion(val_outputs, y_test_tensor).item()
            
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        
        if (epoch+1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
    # 5. Evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        _, predicted = torch.max(test_outputs.data, 1)
        correct = (predicted == y_test_tensor).sum().item()
        accuracy = correct / len(y_test_tensor)
        print(f'\nFinal Test Accuracy: {accuracy * 100:.2f}%')
        
    # 6. Save Model and Plot
    os.makedirs('./saved_model', exist_ok=True)
    torch.save(model.state_dict(), './saved_model/symptom_net.pth')
    print("Model saved to ./saved_model/symptom_net.pth")
    
    # Plot Loss Curve
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss_curve.png')
    print("Loss curve saved to training_loss_curve.png")

if __name__ == "__main__":
    train_model()
