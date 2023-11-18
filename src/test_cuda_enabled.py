import torch

print("Is cuda available? - ", torch.cuda.is_available())
x = torch.rand(5, 3)
print("Simple test: ", x)

