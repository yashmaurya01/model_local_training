import torch
import torch.nn as nn
from datetime import datetime
from syftbox.lib import Client
from torch.utils.data import DataLoader, TensorDataset
import os
from pathlib import Path
import torch.optim as optim
import re


API_NAME = "model_local_training"
SAMPLE_TRAIN_DATASET_DIR = Path("./mnist_samples")
TRAIN_EPOCHS = 10


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_api_private_data(client: Client, api_name: str = API_NAME) -> Path:
    """
    Returns the private data directory of the app
    """
    return client.workspace.data_dir / "private" / api_name


def get_running_folder(client: Client) -> Path:
    """
    Returns the running folder of the app
    """
    return client.api_data(API_NAME) / "running"


def create_private_data_folder(client: Client) -> None:
    """
    Create the private data directory for the api
    where the private test data will be stored according 
    to the following structure:
    ```
    workspace
    ├── apis
    ├── datasites
    ├── logs
    ├── plugins
    ├── private
        ├── model_local_training
        │   └── mnist_label_0.pt  # need to be manually put here by the participant
        │   └── mnist_label_1.pt  # need to be manually put here by the participant
        ...
    ```
    """
    app_pvt_dir = get_api_private_data(client, API_NAME)
    app_pvt_dir.mkdir(parents=True, exist_ok=True)


def init_model_local_training_api(client: Client) -> None:
    """
    Creates the `model_local_training` folder in the `api_data` folder
    with the following structure:
    ```
    datasites
    ├── client_email
    │   └── api_data
            └── model_local_training
                    └── running
    ```
    """
    create_private_data_folder(client)


def look_for_datasets(client: Client) -> list:
    """
    Look for the datasets in the `workspace/private` folder
    """
    prv_dataset_dir: Path = get_api_private_data(client, API_NAME)

    dataset_path_files = prv_dataset_dir.glob("mnist_label_*.pt")
    return [Path(f) for f in dataset_path_files]


def get_public_folder(client: Client) -> Path:
    """
    Returns the public folder of the app
    """
    return client.datasite_path / "public"


def get_model_files(path: Path) -> list[Path]:
    return list(path.glob("trained_mnist_label_*.pt"))


def train_model(dataset_file: Path, output_model_path: Path) -> None:
    start_msg = f"[{datetime.now().isoformat()}] Starting training on {dataset_file.name}...\n"
    print(start_msg)

    images, labels = torch.load(dataset_file)
    dataset = TensorDataset(images, labels)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # model, loss func and optimizer
    model = SimpleNN()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # training loop
    for epoch in range(TRAIN_EPOCHS):
        running_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            # accumulate loss
            running_loss += loss.item()

        # Calculate average loss for the epoch
        avg_loss = running_loss / len(train_loader)
        log_msg = f"[{datetime.now().isoformat()}] Epoch {epoch + 1:04d}: Loss = {avg_loss:.6f}\n"
        print(log_msg)

    torch.save(model.state_dict(), output_model_path)
    print(f"Model saved at {output_model_path}")

    return None


def train_models(client: Client, dataset_paths: list[Path]):
    # handling edge cases before training
    if len(dataset_paths) == 0:
        print("No dataset found in the private folder. Skipping training.")
        return
    
    public_folder: Path = get_public_folder(client)
    output_model_paths: list[Path] = get_model_files(public_folder)
    if len(output_model_paths) == len(dataset_paths):
        print(f"All trained models already exists. Skipping training.")
        return

    for dataset_file in dataset_paths:
        output_model_name = "trained_mnist_label_" + dataset_file.name.split("_")[2].split(".")[0] + ".pt"
        train_model(dataset_file, public_folder / output_model_name)
    
    # print completion message
    final_msg = f"[{datetime.now().isoformat()}] Training on {len(dataset_paths)} datasets completed.\n"
    print(final_msg)


if __name__ == "__main__":
    client = Client.load()

    # Step 1: Initialize the model_local_training API
    init_model_local_training_api(client)

    # Step 2: Look for datasets in the private folder
    dataset_files: list[Path] = look_for_datasets(client)

    # Step 3: Train the models
    train_models(client, dataset_files)
