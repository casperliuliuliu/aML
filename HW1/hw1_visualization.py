
from sklearn import datasets 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

circles_data, circles_data_labels = datasets.make_circles(n_samples=100, factor=0.1, noise=0.1) 

X = torch.Tensor(circles_data)
y = torch.Tensor(circles_data_labels).view(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

print("Training set size:", len(train_dataset))
print("Test set size:", len(test_dataset))

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

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

f0_min, f0_max = circles_data[:, 0].min()-0.5, circles_data[:, 0].max()+0.5
f1_min, f1_max = circles_data[:, 1].min()-0.5, circles_data[:, 1].max()+0.5

def plot_prediction_with_dataloader_continuely(model, dataloader, num_channel, epoch, epochs):
    
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

    x_min = f0_min
    x_max = f0_max
    y_min = f1_min
    y_max = f1_max
    print(x_min)
                         
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))

    # Predict over the mesh grid
    model.eval()
    with torch.no_grad():
        Z = model(torch.Tensor(np.c_[xx.ravel(), yy.ravel()]))
        Z = Z.argmax(dim=1)
        # Z = Z[:,0]
        # Z = torch.sigmoid(Z)

        Z = Z.reshape(xx.shape)

    # Plot the contour and training examples
    plt.figure(figsize=(8, 8))
    plt.contourf(xx, yy, Z, alpha=0.5, colors=['orange','blue', 'yellow', 'red'])

    # contour = plt.contourf(xx, yy, Z, alpha=0.7, levels=np.linspace(Z.min(), Z.max(), 100), cmap='RdYlBu')
    # plt.colorbar(contour)  # Add a colorbar to interpret the values

    plt.scatter(features[labels == 0, 0], features[labels == 0, 1], c='blue', label='Label 0')
    plt.scatter(features[labels == 1, 0], features[labels == 1, 1], c='red', label='Label 1')

    plt.title(f'MODEL with [{num_channel}] hidden layers\nTraining: {(epoch+1)*100//epochs:>2}%')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.xlim(f0_min, f0_max)
    plt.ylim(f1_min, f1_max)
    plt.legend()
    plt.draw()
    plt.pause(0.1)  # Pause for 0.1 second
    plt.clf()  # Clear the plot for the next iteration

small = 1
medium = 7
large = 49
extra = 343

def train_model(num_channel):
    model = SimpleNN(num_channel)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    epochs = 1000
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels[:,0].long())
            loss.backward()
            optimizer.step()

        if epoch % 10 == 9 or epoch < 10:
            plot_prediction_with_dataloader_continuely(model, train_loader, num_channel, epoch, epochs)
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

for ii in range(100,101): 
    train_model(ii)