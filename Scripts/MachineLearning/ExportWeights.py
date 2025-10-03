import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

class MNISTNet(nn.Module):
    def __init__(self, num_hidden=256, dropout=0.2):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, num_hidden)
        self.bn1 = nn.BatchNorm1d(num_hidden)
        self.fc2 = nn.Linear(num_hidden, 10)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)

        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = self.flatten(x)
        x = self.activation(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        return self.fc2(x)

def get_device():
    if torch.cuda.is_available():
        try:
            _ = torch.cuda.memory_allocated()
            print("Using CUDA")
            return torch.device("cuda")
        except RuntimeError:
            print("CUDA error — fallback to CPU")
    return torch.device("cpu")

device = get_device()

def get_data_loaders(batch_size=128):
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST("./data", train=False, download=True, transform=test_transform)

    pin_memory = device.type == "cuda"
    num_workers = os.cpu_count() // 2 if pin_memory else 0

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True, pin_memory=pin_memory, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000,
                                              pin_memory=pin_memory, num_workers=num_workers)
    return train_loader, test_loader

train_loader, test_loader = get_data_loaders()

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, criterion):
    model.eval()
    correct, total, total_loss = 0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader), 100.0 * correct / total

def train_model(epochs=10, patience=7, save_path="best_mnist_model.pth"):
    model = MNISTNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=3)

    best_acc, patience_count = 0.0, 0
    history = {"train_loss": [], "test_loss": [], "test_accuracy": []}

    for epoch in range(epochs):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        test_loss, acc = evaluate(model, test_loader, criterion)
        scheduler.step(test_loss)

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["test_accuracy"].append(acc)

        print(f"Epoch {epoch+1}: Train {train_loss:.4f}, Test {test_loss:.4f}, Acc {acc:.2f}% "
              f"({time.time()-t0:.2f}s)")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), save_path)
            patience_count = 0
            print(f"  Improved to {acc:.2f}%, model saved.")
        else:
            patience_count += 1
            if patience_count >= patience:
                print("Early stopping.")
                break

    model.load_state_dict(torch.load(save_path, map_location=device))
    print(f"Training complete. Best accuracy: {best_acc:.2f}%")
    return model, history, best_acc

def save_weights_to_lua(model, filename="Weights.lua", accuracy=0):
    model.eval()
    w1, b1 = model.fc1.weight.data.cpu().numpy(), model.fc1.bias.data.cpu().numpy()
    w2, b2 = model.fc2.weight.data.cpu().numpy(), model.fc2.bias.data.cpu().numpy()
    bn = model.bn1

    mean, var = bn.running_mean.cpu().numpy(), bn.running_var.cpu().numpy()
    gamma, beta, eps = bn.weight.detach().cpu().numpy(), bn.bias.detach().cpu().numpy(), bn.eps
    scale = gamma / np.sqrt(var + eps)

    w1_folded = (w1 * scale[:, None]).T
    b1_folded = (b1 - mean) * scale + beta
    w2, b2 = w2.T, b2

    with open(filename, "w") as f:
        f.write(f"-- Weights Export (Acc: {accuracy:.2f}%)\nlocal Weights = {{}}\n\n")

        f.write("Weights.weightsIH = {\n")
        for row in w1_folded: f.write("  {" + ",".join(f"{v:.8f}" for v in row) + "},\n")
        f.write("}\n\n")

        f.write("Weights.weightsHO = {\n")
        for row in w2: f.write("  {" + ",".join(f"{v:.8f}" for v in row) + "},\n")
        f.write("}\n\n")

        f.write("Weights.biasH = {" + ", ".join(f"{v:.8f}" for v in b1_folded) + "}\n\n")
        f.write("Weights.biasO = {" + ", ".join(f"{v:.8f}" for v in b2) + "}\n\n")
        f.write("return Weights")

    print(f"Saved weights to {filename}")

if __name__ == "__main__":
    model, history, best_acc = train_model(epochs=1)
    save_weights_to_lua(model, accuracy=best_acc)
