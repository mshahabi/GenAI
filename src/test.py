import torch
import math

print(torch.backends.mps.is_available()) #the MacOS is higher than 12.3+
print(torch.backends.mps.is_built()) #MPS is activated


import torch
import torch.nn as nn
import torch.optim as optim
import time

# Check if GPU is available
# device = torch.device("mps" if torch.cuda.is_available() else "cpu")
device = torch.device("mps")
# device = torch.device("cpu")

print(device)
