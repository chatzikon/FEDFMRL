import torchvision.models
import torch
from resnet import *

model_a = torchvision.models.resnet50(pretrained=True, progress=True)
model_a.fc = torch.nn.Linear(model_a.fc.in_features,100)


model_b= resnet(n_classes=100, depth=56)


print(sum(p.numel() for p in model_a.parameters()))
print(sum(p.numel() for p in model_b.parameters()))

print(sum(p.numel() for p in model_a.parameters())/sum(p.numel() for p in model_b.parameters()))
