import io
import os
import time
import requests
import tarfile
from tqdm import tqdm

import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
import torchvision.models as models
import torchvision.transforms as transforms


import warnings
warnings.simplefilter('ignore')

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available")
else:
    device = torch.device("cpu")
    print("CUDA is not available, using CPU instead")

# check for the cinic-10 data folder
if os.path.exists("community-examples/cinic-10-data"):
    print("CINIC-10 data folder already exists")
else:
    # Define the URL for the dataset file
    data_url = "https://datashare.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz?sequence=4&isAllowed=y"

    # Create the data directory if it doesn't exist
    data_dir = "cinic-10-data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Download the dataset file
    print("Downloading CINIC-10 dataset...")
    data_file = os.path.join(data_dir, "CINIC-10.tar.gz")

    response = requests.get(data_url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024

    start_time = time.time()
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)

    with open(data_file, 'wb') as f:
        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:
                f.write(chunk)
                progress_bar.update(len(chunk))

    end_time = time.time()
    download_time = end_time - start_time
    progress_bar.close()

    print(f"\nDownload time: {download_time:.2f} seconds")

    # Extract the dataset files
    print("Extracting dataset files...")
    with tarfile.open(data_file, 'r:gz') as tar:
        tar.extractall(path=data_dir)

    print("Dataset downloaded and extracted successfully!")

# Define the image classes
classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# transformation function 
transform_train = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((32, 32)),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((32, 32)),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


lr = 1e-3
momentum = 0.9
number_of_epochs = 3
cinic_root = "community-examples/cnn-model-with-cinic-10/cinic-10-data"
train_dataset_path = "community-examples/cnn-model-with-cinic-10/cinic-10-data/cinic-10-data_train.lance/"
test_dataset_path = "community-examples/cnn-model-with-cinic-10/cinic-10-data/cinic-10-data_test.lance/"
validation_dataset_path = "community-examples/cnn-model-with-cinic-10/cinic-10-data/cinic-10-data_valid.lance/"
model_batch_size = 64
batches_to_train = 256
batches_to_val = 128


class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.resnet = models.resnet34(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)
    

def train_model(train_loader, val_loader, model, criterion, optimizer, device, num_epochs, batch_to_train, batch_to_val, run_name):
    wandb.init(project="cinic-10", name = run_name)
    
    model.train()
    total_start = time.time()

    for epoch in range(num_epochs):
        running_loss = 0.0
        total_batch_start = time.time()

        with tqdm(enumerate(train_loader), total=batch_to_train, desc=f"Epoch {epoch+1}") as pbar_epoch:
            for i, data in pbar_epoch:
                if i >= batch_to_train:
                    break

                optimizer.zero_grad()
                inputs, labels = data[0].to(device), data[1].to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if i % 10 == 0:
                    pbar_epoch.set_postfix({'Loss': loss.item()})
                    pbar_epoch.update(10)

        per_epoch_time = (time.time() - total_batch_start) / 60
        avg_loss = running_loss / batch_to_train
        print(f'Epoch {epoch+1}/{num_epochs} | Avg Loss: {avg_loss:.4f} | Time: {per_epoch_time:.4f} mins')
        wandb.log({"Loss": loss.item()})
        wandb.log({"Epoch Duration": per_epoch_time / 60})

    total_training_time = (time.time() - total_start) / 60
    print(f"Total Training Time: {total_training_time:.4f} mins")


    # Validation
    model.eval()
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            if i >= batch_to_val:
                break
            images_val, labels_val = data[0].to(device), data[1].to(device)
            outputs_val = model(images_val)
            _, predicted_val = torch.max(outputs_val.data, 1)
            total_val += labels_val.size(0)
            correct_val += (predicted_val == labels_val).sum().item()

    val_accuracy = 100 * correct_val / total_val
    print(f'Validation Accuracy: {val_accuracy:.2f}%')
    wandb.log({"Validation Accuracy": val_accuracy})
    print('Finished Training')
    return model


vanilla_train_dataset = ImageFolder(root=f'{cinic_root}/train', transform=transform_train)
vanilla_test_dataset = ImageFolder(root=f'{cinic_root}/test', transform=transform_test)
vanilla_val_dataset = ImageFolder(root=f'{cinic_root}/valid', transform=transform_val)

vanilla_train_loader = torch.utils.data.DataLoader(vanilla_train_dataset, batch_size=model_batch_size, shuffle=True)
vanilla_test_loader = torch.utils.data.DataLoader(vanilla_test_dataset, batch_size=model_batch_size, shuffle=True)
vanilla_val_loader = torch.utils.data.DataLoader(vanilla_val_dataset, batch_size=model_batch_size, shuffle=True)

net = Net(len(classes)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

vanilla_trained_model = train_model(vanilla_train_loader, vanilla_val_loader, net, criterion, optimizer, device, number_of_epochs, batches_to_train, batches_to_val, run_name = "vanilla_run_testing")

DIR_PATH = 'community-examples/cnn-model-with-cinic-10/'

# Define the file paths
PATH_VANILLA_MODEL = os.path.join(DIR_PATH, 'cinic_resnet_vanilla_model.pth')

# Check if the directory exists or not, if not create it
if not os.path.isdir(DIR_PATH):
    os.mkdir(DIR_PATH)

# Save the model
torch.save(vanilla_trained_model.state_dict(), PATH_VANILLA_MODEL)

def test_model(test_loader, model, device, type):
    model.eval()
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for data in tqdm(test_loader, desc=f"Testing {type}"):
            images_test, labels_test = data[0].to(device), data[1].to(device)
            outputs_test = model(images_test)
            _, predicted_test = torch.max(outputs_test.data, 1)
            total_test += labels_test.size(0)
            correct_test += (predicted_test == labels_test).sum().item()

    test_accuracy = 100 * correct_test / total_test
    print(f'Test Accuracy: {test_accuracy:.2f}% for {type} dataloader')

test_model(vanilla_test_loader, vanilla_trained_model, device, "vanilla")