# %%
from sklearn import datasets 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

circles_data, circles_data_labels = datasets.make_circles(n_samples=100, factor=0.1, noise=0.1) 


# %%

def plot_data(features, labels):
    blue_points = features[labels == 0]
    red_points = features[labels == 1]
    
    plt.scatter(blue_points[:, 0], blue_points[:, 1], color='blue', label='Label 0')
    plt.scatter(red_points[:, 0], red_points[:, 1], color='red', label='Label 1')
    
    plt.title('Dataset with 2 Features')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

plot_data(circles_data, circles_data_labels)

# %%
X = torch.Tensor(circles_data)
y = torch.Tensor(circles_data_labels).view(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

print("Training set size:", len(train_dataset))
print("Test set size:", len(test_dataset))

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


# %% [markdown]
# # Model Structure Define

# %%
class SimpleNN(nn.Module):
    def __init__(self, channel):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, channel)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(channel, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def plot_prediction_with_dataloader_continuely(model, dataloader):
    
    global f0_max, f0_min, f1_max, f1_min
    all_features = []
    all_labels = []
    
    # Iterate over the DataLoader to accumulate all features and labels
    for batch_features, batch_labels in dataloader:
        # Assuming features and labels are tensors, adjust if necessary
        all_features.append(batch_features.numpy())
        all_labels.append(batch_labels.numpy())

    # Concatenate all features and labels
    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0).flatten()
    print(type(features))
    print(type(labels))
    print(features.shape)
    print(labels.shape)

    
    # Generate a mesh grid to plot decision boundary
    # x_min, x_max = features[:, 0].min() - 0.1, features[:, 0].max() + 0.1
    # y_min, y_max = features[:, 1].min() - 0.1, features[:, 1].max() + 0.1
    x_min = f0_min
    x_max = f0_max
    y_min = f1_min
    y_max = f1_max
    print(x_min)
                         
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    # Predict over the mesh grid
    model.eval()
    with torch.no_grad():
        Z = model(torch.Tensor(np.c_[xx.ravel(), yy.ravel()]))
        Z = Z.argmax(dim=1)
        Z = Z.reshape(xx.shape)

    # print(labels == 0)
    # Plot the contour and training examples
    plt.figure(figsize=(8, 8))
    plt.contourf(xx, yy, Z, alpha=0.5, colors=['orange','blue', 'yellow', 'red'])
    plt.scatter(features[labels == 0, 0], features[labels == 0, 1], c='blue', label='Label 0')
    plt.scatter(features[labels == 1, 0], features[labels == 1, 1], c='red', label='Label 1')

    plt.title('Dataset with Model Predictions')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.xlim(f0_min, f0_max)
    plt.ylim(f1_min, f1_max)
    plt.legend()
    plt.draw()
    plt.pause(0.1)  # Pause for 0.1 second
    plt.clf()  # Clear the plot for the next iteration
# %%
model = SimpleNN(7)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 100
for epoch in range(epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels[:,0].long())
        loss.backward()
        optimizer.step()

    if epoch % 10 == 9:
        plot_prediction_with_dataloader_continuely(model, train_loader)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')


# # %%
# def evaluate_model(model, test_loader):
#     model.eval()
#     correct = 0
#     total = 0

#     with torch.no_grad():
#         for inputs, labels in test_loader:
#             outputs = model(inputs)
#             predicted = (outputs.data > 0.5).float()  # Convert probabilities to binary predictions
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#     accuracy = correct / total
#     return accuracy

# test_accuracy = evaluate_model(model, test_loader)
# print(f'Test Accuracy: {test_accuracy * 100:.2f}%')


# # %%
# f0_min, f0_max = 0, 0 
# f1_min, f1_max = 0, 0
# def plot_data_with_predictions(model, features, labels):
#     global f0_max, f0_min, f1_max, f1_min
#     f0_min, f0_max = features[:, 0].min()-0.5, features[:, 0].max()+0.5
#     f1_min, f1_max = features[:, 1].min()-0.5, features[:, 1].max()+0.5
#     f0, f1 = np.meshgrid(np.arange(f0_min, f0_max, 0.01),
#                          np.arange(f1_min, f1_max, 0.01))

#     inputs = torch.Tensor(np.c_[f0.ravel(), f1.ravel()])
                         
#     model.eval()
#     with torch.no_grad():
#         outputs = model(inputs)
#         outputs = outputs.argmax(dim=1)
#         outputs = outputs.reshape(f0.shape)

#     plt.figure(figsize=(8, 8))
        
#     plt.contourf(f0, f1, outputs, alpha=0.5, colors=['orange','blue', 'yellow', 'red'])

#     blue_points = features[labels == 0]
#     red_points = features[labels == 1]

#     plt.scatter(blue_points[:, 0], blue_points[:, 1], color='blue', label='Label 0')
#     plt.scatter(red_points[:, 0], red_points[:, 1], color='red', label='Label 1')
    
#     plt.title('Dataset with Model Predictions')
#     plt.xlabel('Feature 1')
#     plt.ylabel('Feature 2')
#     plt.xlim(f0_min, f0_max)
#     plt.ylim(f1_min, f1_max)
#     plt.legend()
#     plt.show()

# plot_data_with_predictions(model, circles_data, circles_data_labels)
# print(f0_min, f0_max, f1_min, f1_max)


# # %%
# def plot_data_with_predictions(model, features, labels):

#     global f0_max, f0_min, f1_max, f1_min
#     x_min = f0_min
#     x_max = f0_max
#     y_min = f1_min
#     y_max = f1_max

#     xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
#                          np.linspace(y_min, y_max, 500))
#     print(xx.shape, yy.shape)

#     # Predict over the mesh grid
#     model.eval()
#     with torch.no_grad():
#         Z = model(torch.Tensor(np.c_[xx.ravel(), yy.ravel()]))
#         Z = Z[:,0]
#         Z = torch.sigmoid(Z)
#         Z = Z.reshape(xx.shape)

#     # Plot the contour and training examples
#     plt.figure(figsize=(8, 6))
#     # Use a continuous colormap to reflect probabilities or model scores
#     contour = plt.contourf(xx, yy, Z, alpha=0.5, levels=np.linspace(Z.min(), Z.max(), 100), cmap='RdYlBu')
#     plt.colorbar(contour)  # Add a colorbar to interpret the values

#     # Add a black line where Z == 0.5, indicating the decision boundary
#     contour_lines = plt.contour(xx, yy, Z, levels=[0.5], colors='black')
#     plt.clabel(contour_lines, inline=True, fontsize=8, fmt='0.5')
    
#     # Plot the original data points with labels
#     plt.scatter(features[labels == 0, 0], features[labels == 0, 1], c='blue', label='Label 0')
#     plt.scatter(features[labels == 1, 0], features[labels == 1, 1], c='red', label='Label 1')

#     plt.title('Dataset with Model Predictions')
#     plt.xlabel('Feature 1')
#     plt.ylabel('Feature 2')
#     plt.xlim(f0_min, f0_max)
#     plt.ylim(f1_min, f1_max)
#     plt.legend()
#     plt.show()

# plot_data_with_predictions(model, circles_data, circles_data_labels)


# # %%
# def plot_prediction_with_dataloader(model, dataloader):
    
#     global f0_max, f0_min, f1_max, f1_min
#     all_features = []
#     all_labels = []
    
#     # Iterate over the DataLoader to accumulate all features and labels
#     for batch_features, batch_labels in dataloader:
#         # Assuming features and labels are tensors, adjust if necessary
#         all_features.append(batch_features.numpy())
#         all_labels.append(batch_labels.numpy())

#     # Concatenate all features and labels
#     features = np.concatenate(all_features, axis=0)
#     labels = np.concatenate(all_labels, axis=0).flatten()
#     print(type(features))
#     print(type(labels))
#     print(features.shape)
#     print(labels.shape)

    
#     # Generate a mesh grid to plot decision boundary
#     # x_min, x_max = features[:, 0].min() - 0.1, features[:, 0].max() + 0.1
#     # y_min, y_max = features[:, 1].min() - 0.1, features[:, 1].max() + 0.1
#     x_min = f0_min
#     x_max = f0_max
#     y_min = f1_min
#     y_max = f1_max
#     print(x_min)
                         
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
#                          np.arange(y_min, y_max, 0.01))

#     # Predict over the mesh grid
#     model.eval()
#     with torch.no_grad():
#         Z = model(torch.Tensor(np.c_[xx.ravel(), yy.ravel()]))
#         Z = Z.argmax(dim=1)
#         Z = Z.reshape(xx.shape)

#     # print(labels == 0)
#     # Plot the contour and training examples
#     plt.figure(figsize=(8, 8))
#     plt.contourf(xx, yy, Z, alpha=0.5, colors=['orange','blue', 'yellow', 'red'])
#     plt.scatter(features[labels == 0, 0], features[labels == 0, 1], c='blue', label='Label 0')
#     plt.scatter(features[labels == 1, 0], features[labels == 1, 1], c='red', label='Label 1')

#     plt.title('Dataset with Model Predictions')
#     plt.xlabel('Feature 1')
#     plt.ylabel('Feature 2')
#     plt.xlim(f0_min, f0_max)
#     plt.ylim(f1_min, f1_max)
#     plt.legend()
#     plt.show()

# plot_prediction_with_dataloader(model, train_loader)


# # %%




