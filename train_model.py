import argparse
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.optim as optim  # type: ignore
import torchvision.transforms as transforms  # type: ignore
from torchvision.datasets import CIFAR10  # type: ignore
from torch.utils.data import DataLoader  # type: ignore
from models.resnet import ResNet18_Slim
import binaryconnect
import wandb # type: ignore
from thop import profile # type: ignore

# Argparse arguments + fixed arguments
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.01, help="Learning Rate")
parser.add_argument("--name", type=str, default="resnet18", help="Model Name")
parser.add_argument("--bs", type=int, default=32, help="Batch Size")
parser.add_argument("--e", type=int, default=20, help="Number of epochs")
args = parser.parse_args()

batch_size = args.bs
lr = args.lr
name = args.name
train_epochs = args.e

# Device setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# WandB setup
wandb.init(project="cifar10-training", config={
    "learning_rate" : lr,
    "batch_size": batch_size,
})
wandb.run.name = f"{name}"
wandb.run.save()

# Data augmentation
normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    normalize
])
trainset = CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = CIFAR10(root='./data', train=False, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=batch_size)

# Model setup
model = ResNet18_Slim(use_maxpool=True, use_1x1_conv=False).to(device)
model.eval()

# Optimizer + Criterion + Scheduler
optimizer = optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_epochs)
criterion = nn.CrossEntropyLoss()

for epoch in range(train_epochs):
    # Training
    print(f"Epoch {epoch + 1}/{train_epochs}")
    running_loss = 0.0
    model.train()

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()

        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    scheduler.step()

    # Evaluation
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model.forward(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Loss: {running_loss / len(trainloader):.4f}, Test Accuracy: {accuracy:.2f}%")
    wandb.log({"epoch": epoch+1, "accuracy": accuracy, "loss": running_loss})

# Final evaluation 
correct = 0
total = 0
model.eval()
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model.forward(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
accuracy = 100 * correct / total
print(f"Final accuracy: {accuracy}%")
wandb.log({"final_accuracy": accuracy})

# Thop profile
for data in trainloader:
    inputs, labels = data[0].to(device), data[1].to(device)
    single_input = inputs[0].unsqueeze(0)
    macs, params = profile(model, inputs=(single_input,))
    break

print(f"Nombre de MACs : {macs:,}")
print(f"Nombre de paramètres : {params:,}")
# wandb.log({"MAC": macs, "Parameters": params})
