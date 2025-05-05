import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader

# 1. Set device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# 2. Define transforms for the data
transform = transforms.Compose([
    transforms.Resize(224),  # 224x224
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet mean
                         [0.229, 0.224, 0.225])  # ImageNet std
])

# 3. Load CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset  = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

# Creates data loaders for easier management of image files for shuffling/batching
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 4. Load pretrained ResNet18
model = models.resnet18(pretrained=True)

# 5. Replace the final layer for 10 classes
model.fc = nn.Linear(model.fc.in_features, 10)

# Move the model to the selected device (CPU or GPU)
model = model.to(device)

# 6. Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 7. Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Predict outputs and calculate loss
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()    # clear previous gradients
        loss.backward()    # Calculate new gradients
        optimizer.step()   # Update weights

        total_loss += loss.item()    # Calculate total loss per example

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

# 8. Evaluate on test data
model.eval()   # Switch to evaluation mode (disables dropout, etc.)
correct = 0
total = 0

# Disable gradient calculation for evaluation (faster, uses less memory)
with torch.no_grad():    
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")
