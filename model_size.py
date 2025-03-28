import argparse
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
import torch.nn.utils.prune as prune # type: ignore
import torchvision.transforms as transforms # type: ignore
from torchvision.datasets import CIFAR10 # type: ignore
from torch.utils.data import DataLoader # type: ignore
from models.resnet import ResNet18, ResNet18_Slim, ResNet18_Test
from thop import profile # type: ignore

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="run", help="Model Name")
args = parser.parse_args()
name = args.name

# Device setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data transformation
normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
transform = transforms.Compose([transforms.ToTensor(), normalize])
trainset = CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

# Model setup
model = ResNet18_Slim().to(device)
model.load_state_dict(torch.load(f"{name}.pth", map_location=device))
model.eval()

# Profiling
for data in trainloader:
    inputs, labels = data[0].to(device), data[1].to(device)
    
    single_input = inputs[0].unsqueeze(0)
    
    macs, params = profile(model, inputs=(single_input,))
    break

print(f"Nombre de MACs : {macs:,}")
print(f"Nombre de paramètres : {params:,}")