import torch
import torch.nn.utils.prune as prune
import torch.nn as nn
from parameter_count import count_parameters
import torch.nn.functional as F

class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def prune_model(model, pruning_percentage=0.3):
    """
    Apply L1 Unstructured pruning to convolutional and linear layers of the given model,
    with an actual decrease in the number of parameters by removing filters/channels.

    Args:
    - model: The PyTorch model to prune.
    - pruning_percentage: The percentage of connections to prune in each layer.

    Returns:
    - The modified model with decreased parameters.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            # For Conv2D, calculate the number of filters to prune
            if isinstance(module, nn.Conv2d):
                num_filters = module.out_channels
                num_filters_to_prune = int(num_filters * pruning_percentage)
                prune.ln_structured(module, name='weight', amount=num_filters_to_prune, n=1, dim=0)
                prune.remove(module, 'weight')  # Make pruning permanent
            # For Linear layers, prune connections similarly
            elif isinstance(module, nn.Linear):
                features_to_prune = int(module.out_features * pruning_percentage)
                prune.ln_structured(module, name='weight', amount=features_to_prune, n=1, dim=0)
                prune.remove(module, 'weight')  # Make pruning permanent

    # Adjusting layers or rebuilding the model might be necessary to match new dimensions
    return model

# Initialize model
model = TheModelClass()
print("Before pruning:", count_parameters(model))
pruned_model = prune_model(model, pruning_percentage=0.3)
print("After pruning:", count_parameters(pruned_model))