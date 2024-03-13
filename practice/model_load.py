import torch
model = torch.jit.load('model_scripted.pt')
print(model)
model.eval()