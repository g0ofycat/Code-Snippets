import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import time
import os
from pathlib import Path

class MNISTNet(nn.Module):
    def __init__(self, num_hidden=256):
        super(MNISTNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, num_hidden)
        self.bn1 = nn.BatchNorm1d(num_hidden)

        self.activation = nn.ReLU() 
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(num_hidden, 10)

        self._init_weights()

    def _init_weights(self):
        if isinstance(self.activation, nn.ReLU):
            nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        else:
            nn.init.xavier_normal_(self.fc1.weight)

        nn.init.zeros_(self.fc1.bias)

        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)

        x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    try:
        torch.cuda.memory_allocated()
        print(f"Using device: {device} (GPU memory available)")
    except RuntimeError as e:
        print(f"CUDA error: {e}. Falling back to CPU.")
        device = torch.device('cpu')
else:
    print(f"Using device: {device}")

train_transform = transforms.Compose([
    transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


try:
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=test_transform)
except Exception as e:
     print(f"Error downloading/loading MNIST dataset: {e}")
     print("Please check your internet connection or disk space.")
     exit()

pin_memory = device.type == 'cuda'

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, pin_memory=pin_memory, num_workers=os.cpu_count()//2 if pin_memory else 0)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, pin_memory=pin_memory, num_workers=os.cpu_count()//2 if pin_memory else 0)

model = MNISTNet(num_hidden=256).to(device) 
criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)

def train(model, train_loader, criterion, optimizer, epoch, device):
    model.train()
    start_time = time.time()
    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (batch_idx + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    epoch_loss = running_loss / len(train_loader)
    epoch_time = time.time() - start_time
    print(f'Epoch {epoch+1} completed in {epoch_time:.2f} seconds. Training Loss: {epoch_loss:.4f}')

    return epoch_loss

def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    start_time = time.time()
    test_loss = 0.0
    criterion_eval = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion_eval(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_test_loss = test_loss / len(test_loader)
    accuracy = 100 * correct / total
    eval_time = time.time() - start_time
    print(f'Test Accuracy: {accuracy:.2f}%, Average Test Loss: {avg_test_loss:.4f} (evaluated in {eval_time:.2f} seconds)')

    return avg_test_loss, accuracy

best_accuracy = 0
best_loss = float('inf')
patience_counter = 0
early_stopping_patience = 7 
epochs = 1 
history = {'train_loss': [], 'test_loss': [], 'test_accuracy': []}
model_path = 'best_mnist_model.pth'

print("Starting Training...")
for epoch in range(epochs):
    train_loss = train(model, train_loader, criterion, optimizer, epoch, device)
    test_loss, accuracy = evaluate(model, test_loader, device)

    scheduler.step(test_loss)

    history['train_loss'].append(train_loss)
    history['test_loss'].append(test_loss)
    history['test_accuracy'].append(accuracy)

    if accuracy > best_accuracy:
        print(f"Accuracy improved ({best_accuracy:.2f}% -> {accuracy:.2f}%). Saving model...")
        best_accuracy = accuracy
        torch.save(model.state_dict(), model_path)
        patience_counter = 0

    else:
        patience_counter += 1
        print(f"No improvement in accuracy for {patience_counter} epoch(s).")

    if patience_counter >= early_stopping_patience:
        print(f"Early stopping triggered after {epoch+1} epochs due to lack of accuracy improvement.")
        break

print(f"Training finished. Best test accuracy: {best_accuracy:.2f}%")

print(f"Loading best weights from {model_path}")
try:
    map_location = torch.device('cpu') if not torch.cuda.is_available() else None
    model.load_state_dict(torch.load(model_path, map_location=map_location))
    print("Successfully loaded best model weights.")
    evaluate(model, test_loader, device)
except FileNotFoundError:
    print(f"Error: '{model_path}' not found. Using final model weights.")
except Exception as e:
    print(f"Error loading model weights: {e}. Using final model weights.")


def save_weights_to_lua(model, filename="Weights.lua"):
    model.eval()

    weights_ih = model.fc1.weight.data.cpu().numpy()  # [256, 784]
    bias_h = model.fc1.bias.data.cpu().numpy()
    bn_running_mean = model.bn1.running_mean.data.cpu().numpy()
    bn_running_var = model.bn1.running_var.data.cpu().numpy()
    bn_weight = model.bn1.weight.data.cpu().numpy()
    bn_bias = model.bn1.bias.data.cpu().numpy()
    bn_eps = model.bn1.eps
    weights_ho = model.fc2.weight.data.cpu().numpy()  # [10, 256]
    bias_o = model.fc2.bias.data.cpu().numpy()

    bn_scale = bn_weight / np.sqrt(bn_running_var + bn_eps)
    weights_ih_folded = weights_ih * bn_scale[:, np.newaxis]
    bias_h_folded = (bias_h - bn_running_mean) * bn_scale + bn_bias

    weights_ih_folded = weights_ih_folded.T 
    weights_ho = weights_ho.T

    with open(filename, 'w') as f:
        f.write(f"-- // Format: [Input][Hidden] for weightsIH, [Hidden][Output] for weightsHO. Accuracy: {best_accuracy:.2f}%\n")
        f.write("local Weights = {}\n\n")

        # Linear Layer 1 (folded) - [Input][Hidden]
        f.write("Weights.weightsIH = { -- // [Input][Hidden]\n")
        for i in range(weights_ih_folded.shape[0]):  # 784 inputs
            f.write("  {")
            f.write(",".join(f"{w:.8f}" for w in weights_ih_folded[i, :]))
            f.write("},\n")
        f.write("}\n\n")

        # Linear Layer 2 - [Hidden][Output]
        f.write("Weights.weightsHO = { -- // [Hidden][Output]\n")
        for j in range(weights_ho.shape[0]):  # 256 hidden
            f.write("  {")
            f.write(",".join(f"{w:.8f}" for w in weights_ho[j, :]))
            f.write("},\n")
        f.write("}\n\n")

        f.write("Weights.biasH = { -- // [Hidden]\n  ")
        f.write(", ".join(f"{b:.8f}" for b in bias_h_folded))
        f.write("\n}\n\n")

        f.write("Weights.biasO = { -- // [Output]\n  ")
        f.write(", ".join(f"{b:.8f}" for b in bias_o))
        f.write("\n}\n\n")

        f.write("return Weights")
    print(f"Finished saving weights to {filename}")


save_weights_to_lua(model)
