import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

width = 1024
height = 1024

xx = torch.arange(0, width, 1, device=device, dtype=torch.float32)
yy = torch.arange(0, height, 1, device=device, dtype=torch.float32)
xxx = xx.expand((height, -1))
yyy = yy.expand((width, -1))
yyy = yyy.transpose(0, 1)

xy = torch.stack([xxx, yyy], dim=1)