import numpy as np
import torch
import torchvision.models as models
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
from datasets import load_dataset
from torch.utils.data import Dataset

class RecyclingDataset(Dataset):
    def __init__(self, hf_dataset):
        self.data = hf_dataset

        self.transform = v2.Compose([
            v2.Resize((52, 52)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        image = example["image"].convert("RGB")
        image = self.transform(image)

        categories = example["objects"]["category"]
        label = categories[0]

        return image, torch.tensor(label, dtype=torch.long)

ds = load_dataset(
    "nflechas/recycling_app",
    revision="refs/convert/parquet"
)

print("---")
print(ds["train"][0])
print("---")

train_dataset = RecyclingDataset(ds["train"]) 
val_dataset = RecyclingDataset(ds["validation"])
test_dataset = RecyclingDataset(ds["test"])

batch_size = 64

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=True
)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(52*52*3, 2048), # affine linear transformation
            nn.ReLU(), # clamp (0, x)
            nn.Linear(2048, 2048), # intermediate step after ReLU trans, 2048 to 2048
            nn.ReLU(),
            nn.Linear(2048, 6) # final output step, maps 2048 to 6 outputs (or classifications?)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("modelV2.pth", weights_only=True))
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_loader, model, loss_fn, optimizer)
    test(test_loader, model, loss_fn)
print("Done!")

torch.save(model.state_dict(), "modelV2.pth")
print("Saved PyTorch Model State to modelV2.pth")

classes = [
    "biodegradable",
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
]

model.eval()
x, y = val_dataset[53]
x = x.unsqueeze(0).to(device)

model.eval()
with torch.no_grad():
    logits = model(x)

predicted_index = logits.argmax(dim=1).item()
actual_index = y.item()

print(
    f"Predicted: {classes[predicted_index]}, "
    f"Actual: {classes[actual_index]}"
)