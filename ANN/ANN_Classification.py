import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

df = pd.read_csv("DateFruit Dataset.csv")
# print(df.head())

X = df.drop("Class", axis = 1)
y = df["Class"]

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------------------------------------------ANN-------------------------------------------------------------------

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# -----------------------------------------------------------------Build Model--------------------------------------------------------------

class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(X.shape[1], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 7)
        )
    
    def forward(self, x):
        return self.model(x)
    
model = ANN()

# ----------------------------Loss & Optim---------------------------------------------------

criteria = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# ----------------------------Training ANN----------------------------------------------------

epochs = 100

for epoch in range(epochs):
    model.train()

    running_loss = 0.0

    for xb, yb in train_loader:
        optimizer.zero_grad()

        outputs = model(xb)
        loss = criteria(outputs, yb)
        loss.backward()
        optimizer.step() # params update

        running_loss += loss.item()
    
    train_loss = running_loss / len(train_loader)

    print(f"epoc = {epoch + 1}/{epochs} ==> loss = {train_loss}")


# ------------------------------------------------------------------------Evaluate-----------------------------------------------------------------------

model.eval()

total = 0
correct = 0

with torch.no_grad():
    for xb, yb in test_loader:
        outputs = model(xb)
        _, predicted = torch.max(outputs, 1) # returns max value and its index

        correct += (predicted == yb).sum().item()
        total += yb.size(0) # actual samples in each batch

print("accuracy: ", correct / total)