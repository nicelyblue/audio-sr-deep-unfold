import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import AudioDataset, fft_transform
from model import UNet
from loss import MagnitudePhaseLoss
from early_stopping import EarlyStopping

def train_model(model, train_loader, val_loader, loss_function, optimizer, num_epochs=10, patience=5):
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, verbose=True)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = loss_function(outputs, targets)
                val_loss += loss.item()

        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        print(f'Epoch {epoch+1}, Train Loss: {train_loss}, Val Loss: {val_loss}')

        scheduler.step(val_loss)
        early_stopping(val_loss)

        if early_stopping.early_stop:
            print("Early stopping")
            break


def test_model(model, test_loader, loss_function):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            test_loss += loss.item()
    test_loss /= len(test_loader)
    print(f'Test Loss: {test_loss}')


def main():
    input_dataset_path = 'path/to/your/input/dataset'
    target_dataset_path = 'path/to/your/target/dataset'

    input_paths = [os.path.join(input_dataset_path, file) for file in sorted(os.listdir(input_dataset_path))]
    target_paths = [os.path.join(target_dataset_path, file) for file in sorted(os.listdir(target_dataset_path))]
    
    assert len(input_paths) == len(target_paths), "Input and target datasets must be the same size"
    
    full_dataset = AudioDataset(input_paths=input_paths, target_paths=target_paths, transform=fft_transform)

    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    model = UNet(n_channels=3, n_classes=3)
    loss_function = MagnitudePhaseLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 500
    train_model(model, train_loader, val_loader, loss_function, optimizer, num_epochs)

    test_model(model, test_loader, loss_function)

if __name__ == "__main__":
    main()
