import argparse
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
import torch.nn.utils.prune as prune # type: ignore
import torchvision.transforms as transforms # type: ignore
from torchvision.datasets import CIFAR10 # type: ignore
from torch.utils.data import DataLoader # type: ignore
from models.resnet import ResNet18_Slim, ResNet18
import binaryconnect # type: ignore
import wandb # type: ignore
from thop import profile # type: ignore

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="run", help="Model Name")
parser.add_argument("--p_s", type=float, default=0, help="Structured pruning percentage (0-100)")
parser.add_argument("--p_u", type=float, default=0, help="Unstructured pruning percentage (0-100)")
parser.add_argument("--global_pruning", type=str, default="True", help="Global pruning (True/False)")
parser.add_argument("--e", type=int, default=20, help="Number of epochs")
args = parser.parse_args()

name = args.name
# Device setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data transformation
normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
transform = transforms.Compose([transforms.ToTensor(), normalize])
trainset = CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = CIFAR10(root='./data', train=False, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64)

# Model setup
model = ResNet18().to(device)
# model.load_state_dict(torch.load(f"saved_models/run1-resnet18.pth", map_location=device)) # type: ignore
model.load_state_dict(torch.load(f"saved_models/run1-resnet18.pth", map_location=device), strict=False)
model.eval()

# Pruning setup
if args.global_pruning == "True":
    if args.p_s > 0:
        raise ValueError("Structured pruning cannot be global_pruning.")
    prune.global_unstructured([(model.conv1, "weight")], pruning_method=prune.L1Unstructured, amount=args.p_u / 100)
else:
    if args.p_u > 0:
        prune.random_unstructured(model.conv1, name="weight", amount=args.p_u / 100)
    if args.p_s > 0:
        prune.ln_structured(model.conv1, name="weight", amount=args.p_s / 100, n=1, dim=3)

# WandB initialization
wandb.init(project="cifar10-training", config={
    "pruning_structured": args.p_s,
    "pruning_unstructured": args.p_u,
})
wandb.run.name = args.name
wandb.run.save()

# Training loop
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()

for epoch in range(args.e):
    running_loss = 0.0
    model.train()
   
    for data in trainloader:
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    # Accuracy evaluation
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
   
    # Log results
    print(f"Epoch : {epoch + 1}")
    print(f"Accuracy : {accuracy}")
    wandb.log({"epoch": epoch + 1, "train_loss": running_loss, "accuracy": accuracy})

# file_name = args.name + ".pth"
# torch.save(model.state_dict(), file_name)
print("Pruning complete.")
